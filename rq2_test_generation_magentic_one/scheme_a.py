#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import sys
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / "core"), str(ROOT / "brs_v0_4"), str(ROOT)]

from adaptive_search import (
    ACTIVE_OBJECTIVE_INDICES,
    CampaignArchive,
    ExecutedTest,
    Gene,
    ModelRuntime,
    OBJECTIVE_DIRECTIONS,
    coverage_features,
    nsga2_complete_path_search,
    update_archive,
)
try:
    from adaptive_search import OBJECTIVE_NAMES as _OBJECTIVE_NAMES
except ImportError:
    _OBJECTIVE_NAMES = None


def active_objective_labels(suffixed=False):
    if _OBJECTIVE_NAMES is None:
        return (["coverage_gain_max", "behavioral_diversity_max", "path_cost_min"]
                if suffixed else
                ["coverage_gain", "behavioral_diversity", "path_cost"])
    rename = {"execution_cost": "path_cost"}
    out = []
    for j in ACTIVE_OBJECTIVE_INDICES:
        base = rename.get(_OBJECTIVE_NAMES[j], _OBJECTIVE_NAMES[j])
        if suffixed:
            base += "_max" if OBJECTIVE_DIRECTIONS[j] > 0 else "_min"
        out.append(base)
    return out


from checkpoint_io import (
    CheckpointRecorder,
    decode_random_state,
    deserialize_archive,
    encode_random_state,
    load_checkpoint,
    sha256_json,
)
from execution_adapter import (
    ModuleExecutor, MockExecutor, extract_llm_usage,
)
from file_provision import (
    apply_hint, resolve_gaia_root, sandbox_hint,
)
from brs_bridge import make_concretizer
from brs_grounding_policy import apply_leave_current_query_out
from brs_user_task_generator import GenerationFailure
from campaign_runner import (
    dependency_for_instance,
    provision_v2_attachments,
)
from llm_client import load_llm

NOT_EVALUATED = {
    "verdict": "NOT_EVALUATED",
    "reason": "Oracle construction is outside the current BRS design.",
}


def genes_from_dicts(rows):
    out = []
    for x in rows or []:
        params = tuple(sorted((str(k), str(v)) for k, v in (x.get("params") or {}).items()))
        out.append(Gene(str(x.get("rule_id", "")), str(x.get("event", "")), params))
    return out


def rank_pareto_by_ideal_active(pareto):
    if not pareto:
        return []
    vals = [x.objectives for x in pareto]
    mins = {j: min(v[j] for v in vals) for j in ACTIVE_OBJECTIVE_INDICES}
    maxs = {j: max(v[j] for v in vals) for j in ACTIVE_OBJECTIVE_INDICES}

    def distance(x):
        d2 = 0.0
        for j in ACTIVE_OBJECTIVE_INDICES:
            v = x.objectives[j]
            lo, hi = mins[j], maxs[j]
            direction = OBJECTIVE_DIRECTIONS[j]
            if hi <= lo:
                z = 1.0
            elif direction > 0:
                z = (v - lo) / (hi - lo)
            else:
                z = (hi - v) / (hi - lo)
            d2 += (1.0 - z) ** 2
        return math.sqrt(d2)

    return sorted(
        [x.clone() for x in pareto],
        key=lambda x: (distance(x), len(x.genes), tuple(g.key for g in x.genes)),
    )


def restore_planning_archive(plans, runtime):
    arc = CampaignArchive()
    for p in plans:
        genes = genes_from_dicts(p["selected_path"])
        arc.covered_features |= coverage_features(runtime, genes)
        arc.executed_tests.append(
            ExecutedTest(list(genes), "", list(genes), "PLANNED_TARGET", [])
        )
    return arc


def phase1_plan(runtime, num_tests, rng, search_kw, concretizer, model,
                max_search_regenerations, nlp_retries, seed_map,
                provision_dir, provision_files, plans_path, is_feasible=None):
    plans = []
    if plans_path.exists():
        saved = json.loads(plans_path.read_text(encoding="utf-8"))
        plans = list(saved.get("plans", []))
        if plans:
            try:
                rng.setstate(decode_random_state(saved["rng_state_b64"]))
            except (KeyError, ValueError):
                plans = []
            else:
                print(
                    f"[schemeA] resumed planning state: {len(plans)} slot(s) "
                    f"already planned from {plans_path.name}",
                    file=sys.stderr,
                )
    planning = restore_planning_archive(plans, runtime)

    for t in range(len(plans) + 1, num_tests + 1):
        chosen = chosen_pf = tc = None
        attempts = []
        search_sec = 0.0
        nlp_sec = 0.0

        for search_attempt in range(1, max_search_regenerations + 1):
            s0 = time.perf_counter()
            nsga_kw = dict(search_kw)
            if is_feasible is not None:
                nsga_kw["is_feasible"] = is_feasible
            pareto, _ = nsga2_complete_path_search(
                runtime, planning, rng, **nsga_kw
            )
            search_sec += time.perf_counter() - s0

            for rank, candidate in enumerate(
                rank_pareto_by_ideal_active(pareto), 1
            ):
                availability = concretizer.can_concretize(candidate.genes, model)
                if not availability.get("ok"):
                    attempts.append({
                        "search_attempt": search_attempt,
                        "pareto_rank": rank,
                        "path": [g.to_dict() for g in candidate.genes],
                        "availability": availability,
                    })
                    continue

                gen_err = None
                g0 = time.perf_counter()
                for gen_attempt in range(1, nlp_retries + 1):
                    try:
                        tc = concretizer.generate(candidate.genes, model, rng)
                        gen_err = None
                        break
                    except GenerationFailure as exc:
                        gen_err = f"{exc.reason}: {exc.details}"
                        if gen_attempt < nlp_retries:
                            time.sleep(2 * gen_attempt)
                nlp_sec += time.perf_counter() - g0
                if gen_err is not None:
                    attempts.append({
                        "search_attempt": search_attempt,
                        "pareto_rank": rank,
                        "path": [g.to_dict() for g in candidate.genes],
                        "availability": availability,
                        "generation_failure": gen_err,
                    })
                    continue

                chosen = candidate
                chosen_pf = pareto
                break
            if chosen is not None:
                break

        if chosen is None:
            reason_counts = Counter(
                (
                    a.get("generation_failure", "").split(":", 1)[0]
                    if a.get("generation_failure")
                    else (a.get("availability") or {}).get("reason", "unknown")
                )
                for a in attempts
            )
            raise RuntimeError(
                f"Scheme A planning: no concretizable path for test slot {t} "
                f"after {max_search_regenerations} search regenerations; "
                f"reasons={dict(reason_counts)}"
            )

        dep = dependency_for_instance(tc, seed_map)
        query_sent = apply_hint(tc.query, dep)
        target_genes = list(chosen.genes)

        if provision_files:
            work_dir = str((provision_dir / f"work_test_{t:02d}").resolve())
        else:
            work_dir = None

        plan = {
            "test_index": t,
            "selected_path": [g.to_dict() for g in target_genes],
            "selected_objectives": {
                "coverage_gain_max": chosen.objectives[0],
                "behavioral_diversity_max": chosen.objectives[1],
                "path_cost_min": chosen.objectives[3],
            },
            "pareto_front": [
                {
                    "path": [g.to_dict() for g in x.genes],
                    "objectives": {
                        "coverage_gain_max": x.objectives[0],
                        "behavioral_diversity_max": x.objectives[1],
                        "neutral_slot": x.objectives[2],
                        "path_cost_min": x.objectives[3],
                    },
                }
                for x in chosen_pf
            ],
            "testcase": asdict(tc),
            "query_sent": query_sent,
            "file_dependency": dep,
            "work_dir": work_dir,
            "search_sec": search_sec,
            "nlp_generation_sec": nlp_sec,
            "concretization_attempts": attempts,
        }
        plans.append(plan)

        planning.covered_features |= coverage_features(runtime, target_genes)
        planning.executed_tests.append(
            ExecutedTest(list(target_genes), "", list(target_genes),
                         "PLANNED_TARGET", [])
        )
        plan["planning_archive_after"] = {
            "selected_target_count": len(planning.executed_tests),
            "target_covered_feature_count": len(planning.covered_features),
        }

        plans_path.write_text(
            json.dumps(
                {
                    "num_tests": num_tests,
                    "plans": plans,
                    "rng_state_b64": encode_random_state(rng.getstate()),
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print(
            f"[schemeA] planned {t}/{num_tests} path_len={len(target_genes)} "
            f"request_frozen len(query)={len(query_sent)}",
            file=sys.stderr,
        )

    return plans


def stage_plan_attachment_inplace(plan):
    dep = plan.get("file_dependency")
    work_dir = plan.get("work_dir")
    plan["staged_attachment"] = None
    if not dep or not work_dir:
        return

    wd = Path(work_dir)
    wd.mkdir(parents=True, exist_ok=True)
    fname = dep.get("file_name") or Path(dep["sandbox_path"]).name
    src = Path(dep.get("sandbox_path") or "")
    if not src.exists():
        src = Path(dep.get("host_path") or "")
    if not src.exists():
        raise FileNotFoundError(
            f"Scheme A staging: attachment missing for test "
            f"{plan.get('test_index')}: {src}")
    dst = wd / fname
    if not dst.exists():
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)

    local_dep = dict(dep)
    local_dep["sandbox_path"] = str(dst.resolve())
    local_dep["hint"] = sandbox_hint(local_dep)
    tc_query = (plan.get("testcase") or {}).get("query") or ""
    if tc_query:
        plan["query_sent"] = apply_hint(tc_query, local_dep)
    plan["file_dependency"] = local_dep
    plan["staged_attachment"] = str(dst.resolve())


def execute_fixed(executor, plan, model, model_path):
    t = plan["test_index"]
    context = {
        "method": "SchemeA-TargetPlanning-BRSv04",
        "test_index": t,
        "model_path": model_path,
        "target_path": genes_from_dicts(plan["selected_path"]),
        "work_dir": plan["work_dir"],
        "file_dependency": plan["file_dependency"],
    }
    t0 = time.perf_counter()
    er = executor.execute(plan["query_sent"], model, context)
    actual = executor.analyze_path(er.raw_log, model)
    return er, actual, time.perf_counter() - t0


def phase2_execute(plans, executor, model, model_path, runtime, rec,
                   resume_done):
    todo = [p for p in plans if p["test_index"] not in resume_done]
    tests = []
    actual_archive = CampaignArchive()

    if resume_done:
        latest = rec.root / "LATEST"
        ck = load_checkpoint(rec.root / latest.read_text(encoding="utf-8").strip())
        actual_archive = deserialize_archive(ck["archive"])
        for k in sorted(resume_done):
            row = load_checkpoint(rec.root / f"C{k:03d}.json").get("last_round")
            if row is not None:
                tests.append(row)
        print(
            f"[schemeA] resume: {len(resume_done)} test(s) already executed; "
            f"running the remaining {len(todo)}",
            file=sys.stderr,
        )
    else:
        rec.save(
            0, actual_archive,
            last_round={"planned_tests": len(plans)},
            search_parameters={}, random_state=None,
        )

    t_start = time.perf_counter()
    for p in todo:
        t = p["test_index"]
        target_genes = genes_from_dicts(p["selected_path"])
        p_exec = p
        try:
            p_exec = dict(p)
            stage_plan_attachment_inplace(p_exec)
            er, actual, wall_sec = execute_fixed(
                executor, p_exec, model, model_path)
        except Exception as exc:
            wall_sec = 0.0
            row = {
                "test_index": t,
                "search_sec": p["search_sec"],
                "nlp_generation_sec": p["nlp_generation_sec"],
                "selected_path": p["selected_path"],
                "selected_objectives": p["selected_objectives"],
                "testcase": p["testcase"],
                "query_sent": p_exec.get("query_sent", p["query_sent"]),
                "file_dependency": p_exec.get("file_dependency", p["file_dependency"]),
                "staged_attachment": p_exec.get("staged_attachment"),
                "actual_path": [],
                "outcome": f"ERROR:ExecutorException",
                "failure_signature": None,
                "final_answer": None,
                "oracle_verdict": NOT_EVALUATED,
                "mas_llm_usage": None,
                "timings": {"exception": f"{type(exc).__name__}: {exc}"},
                "concretization_attempts": p["concretization_attempts"],
                "execution_wall_sec": wall_sec,
            }
            row["planning_archive_after"] = p.get("planning_archive_after")
            row["actual_archive_after"] = {
                "executed_tests": len(actual_archive.executed_tests),
                "actual_covered_feature_count":
                    len(actual_archive.covered_features),
            }
            tests.append(row)
            rec.save(t, actual_archive, last_round=row,
                     search_parameters={}, random_state=None)
            print(f"[schemeA] test {t} EXCEPTION {exc}", file=sys.stderr)
            continue

        update_archive(
            actual_archive, runtime, target_genes, actual,
            p_exec["query_sent"], er.outcome, er.raw_log,
        )
        row = {
            "test_index": t,
            "search_sec": p["search_sec"],
            "nlp_generation_sec": p["nlp_generation_sec"],
            "pareto_front": p["pareto_front"],
            "selected_path": p["selected_path"],
            "selected_objectives": p["selected_objectives"],
            "testcase": p["testcase"],
            "query_sent": p_exec["query_sent"],
            "file_dependency": p_exec["file_dependency"],
            "staged_attachment": p_exec.get("staged_attachment"),
            "actual_path": [g.to_dict() for g in actual],
            "outcome": er.outcome,
            "failure_signature": er.failure_signature,
            "final_answer": er.final_answer,
            "oracle_verdict": NOT_EVALUATED,
            "mas_llm_usage": extract_llm_usage(er.raw_log),
            "timings": er.timings or {},
            "concretization_attempts": p["concretization_attempts"],
            "planning_archive_after": p.get("planning_archive_after"),
            "actual_archive_after": {
                "executed_tests": len(actual_archive.executed_tests),
                "actual_covered_feature_count":
                    len(actual_archive.covered_features),
            },
            "execution_wall_sec": wall_sec,
        }
        tests.append(row)
        rec.save(t, actual_archive, last_round=row,
                 search_parameters={}, random_state=None)
        print(
            f"[schemeA] test {t}/{len(plans)} done "
            f"actual_len={len(actual)} outcome={er.outcome} "
            f"wall={wall_sec:.0f}s",
            file=sys.stderr,
        )

    phase2_sec = time.perf_counter() - t_start
    tests.sort(key=lambda r: r["test_index"])
    return tests, actual_archive, phase2_sec


def run(a):
    model = json.loads(a.model.read_text(encoding="utf-8"))
    runtime = ModelRuntime(model)

    chat = load_llm(a.llm_config, a.llm_id, allow_mock=a.allow_mock)
    concretizer = make_concretizer(
        chat,
        a.temperature,
        a.max_tokens,
        library_root=a.brs_library,
        grounding_mode=a.grounding_mode,
        signature_policy=a.signature_policy,
        n_surface_candidates=a.candidates,
        allow_source_task_fallback=a.allow_source_task_fallback,
    )
    excluded = apply_leave_current_query_out(concretizer, model)

    rq3_variants = {
        "sa_covdiv", "sa_covcost", "sa_divcost",
        "schemea_covdiv", "schemea_covcost", "schemea_divcost",
    }
    variant_key = os.environ.get("MAS_OBJECTIVE_VARIANT", "").strip().lower()
    if variant_key in rq3_variants:
        _semantic_map = concretizer.semantic_map

        def is_feasible(genes):
            return any(
                _semantic_map.classify(g.event, model)["semantic_role"]
                == "operational"
                for g in genes
            )
    else:
        is_feasible = None

    search_kw = dict(
        pop_size=a.pop_size,
        generations=a.generations,
        crossover_rate=a.crossover_rate,
        mutation_rate=a.mutation_rate,
        max_path_len=a.max_path_len,
        path_gen_retries=a.path_gen_retries,
        terminal_template_weight=a.terminal_template_weight,
        k_neighbors=a.k_neighbors,
        alpha=a.alpha,
        beta=a.beta,
    )

    cid = a.campaign_id or (
        f"{model.get('model_id')}_schemeA_{a.llm_id}_seed{a.seed}"
    )

    provision_dir = Path(a.provision_dir)
    gaia_root = resolve_gaia_root(a.gaia_root)
    seed_map = {}
    if a.provision_files:
        seed_map, gaia_root_resolved = provision_v2_attachments(
            concretizer.repo.instances, a.gaia_root, provision_dir,
            a.sandbox_root,
        )
        concretizer.set_attachment_map(seed_map)
        if gaia_root_resolved is None:
            print(
                "[provision] WARNING: GAIA 2023 root not found; file-dependent "
                "tasks will run without attachments",
                file=sys.stderr,
            )
        else:
            print(
                f"[provision] {len(seed_map)} attachment(s) in "
                f"{provision_dir.resolve()} from {gaia_root_resolved}",
                file=sys.stderr,
            )

    rec = CheckpointRecorder(
        a.checkpoint_root, cid, model,
        metadata={
            "method": "SchemeA-TargetPlanning-BRSv04",
            "brs_version": "0.4",
            "llm_id": a.llm_id,
            "seed": a.seed,
            "execution_mode": "sequential",
            "active_objectives": active_objective_labels(),
            "objective_variant": os.environ.get("MAS_OBJECTIVE_VARIANT", "")
                                  or "original",
            "path_validity_constraint": (
                "requires_task_level_operation" if is_feasible is not None
                else "none"),
            "runtime_feedback_used_for_search": False,
            "excluded_grounded_instances": excluded,
            "brs_library": str(a.brs_library),
            "signature_policy": a.signature_policy,
        },
    )
    plans_path = rec.root / "plans.json"

    plan_rng = random.Random(a.seed)
    t_p1 = time.perf_counter()
    print(
        f"[schemeA] Phase 1: planning {a.num_tests} target paths and freezing "
        "requests (no MAS execution)",
        file=sys.stderr,
    )
    plans = phase1_plan(
        runtime, a.num_tests, plan_rng, search_kw, concretizer, model,
        a.max_search_regenerations, a.nlp_retries, seed_map,
        provision_dir, a.provision_files, plans_path, is_feasible,
    )
    phase1_sec = time.perf_counter() - t_p1
    print(f"[schemeA] Phase 1 done in {phase1_sec:.0f}s", file=sys.stderr)

    executor = MockExecutor() if a.executor is None else ModuleExecutor(a.executor)

    resume_done = set()
    if a.resume:
        latest = rec.root / "LATEST"
        if not latest.exists():
            print(
                "[schemeA] --resume: no execution checkpoint found; running all "
                f"{len(plans)} frozen requests from scratch",
                file=sys.stderr,
            )
            latest = None
        if latest is not None:
            ck = load_checkpoint(
                rec.root / latest.read_text(encoding="utf-8").strip()
            )
            if ck.get("model_hash") != sha256_json(model):
                raise SystemExit(
                    "--resume: model file hash differs from the checkpointed "
                    "campaign"
                )
            resume_done = set(range(1, int(ck["test_index"]) + 1))

    print(
        f"[schemeA] Phase 2: executing {a.num_tests - len(resume_done)} "
        "request(s) sequentially (concurrency disabled)",
        file=sys.stderr,
    )
    tests, actual_archive, phase2_sec = phase2_execute(
        plans, executor, model, str(a.model.resolve()), runtime, rec,
        resume_done,
    )
    print(f"[schemeA] Phase 2 done in {phase2_sec:.0f}s", file=sys.stderr)

    out = {
        "method": "SchemeA-TargetPlanning-BRSv04",
        "design": {
            "objective_variant": os.environ.get("MAS_OBJECTIVE_VARIANT", "")
                                 or "original",
            "active_objectives": active_objective_labels(suffixed=True),
            "pareto_selector": "equal-weight normalized ideal point (active only)",
            "planning_archive": "selected target paths",
            "execution_feedback_to_search": False,
            "requests_frozen_before_execution": True,
            "execution_mode": "sequential",
            "path_validity_constraint": (
                "requires_task_level_operation" if is_feasible is not None
                else "none"),
        },
        "brs_version": "0.4",
        "model_id": model.get("model_id"),
        "llm_id": a.llm_id,
        "seed": a.seed,
        "excluded_grounded_instances": excluded,
        "parameters": search_kw,
        "phase_timings_sec": {
            "phase1_planning": round(phase1_sec, 1),
            "phase2_sequential_execution": round(phase2_sec, 1),
            "total_wall": round(phase1_sec + phase2_sec, 1),
        },
        "planning_summary": {
            "planned_tests": len(plans),
            "target_covered_feature_count": len(
                restore_planning_archive(plans, runtime).covered_features
            ),
            "actual_covered_feature_count": len(actual_archive.covered_features),
        },
        "tests": tests,
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return out


def parser():
    p = argparse.ArgumentParser(
        description="Scheme A: target-only 3-objective planning + sequential execution"
    )
    p.add_argument("model", type=Path)
    p.add_argument("--llm-config", type=Path, required=True)
    p.add_argument("--llm-id", required=True)
    p.add_argument("--executor", type=Path, default=None)
    p.add_argument("--allow-mock", action="store_true")
    p.add_argument("--num-tests", type=int, default=30)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--pop-size", type=int, default=100)
    p.add_argument("--generations", type=int, default=200)
    p.add_argument("--crossover-rate", type=float, default=.8)
    p.add_argument("--mutation-rate", type=float, default=.35)
    p.add_argument("--max-path-len", type=int, default=30)
    p.add_argument("--path-gen-retries", type=int, default=30)
    p.add_argument("--terminal-template-weight", type=float, default=.2)
    p.add_argument("--k-neighbors", type=int, default=5)
    p.add_argument("--alpha", type=float, default=1.0,
                   help="compatibility only; realizability is inactive")
    p.add_argument("--beta", type=float, default=1.0,
                   help="compatibility only; realizability is inactive")
    p.add_argument("--max-search-regenerations", type=int, default=3)
    p.add_argument("--temperature", type=float, default=.2)
    p.add_argument("--max-tokens", type=int, default=16000)
    p.add_argument("--candidates", type=int, default=3)
    p.add_argument("--grounding-mode", choices=["bootstrap", "strict"],
                   default="bootstrap")
    p.add_argument("--signature-policy", choices=["exact", "collapse_adjacent"],
                   default="exact")
    p.add_argument("--allow-source-task-fallback", action="store_true")
    p.add_argument("--nlp-retries", type=int, default=5)
    p.add_argument("--brs-library", type=Path,
                   default=ROOT / "brs_v0_4" / "library")
    p.add_argument("--gaia-root", type=Path, default=None)
    p.add_argument("--provision-dir", type=Path,
                   default=ROOT / "results" / "provisioned_files_schemeA")
    p.add_argument("--sandbox-root", default=None)
    p.add_argument("--no-provision-files", dest="provision_files",
                   action="store_false")
    p.set_defaults(provision_files=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--campaign-id", default=None)
    p.add_argument("--checkpoint-root", type=Path,
                   default=ROOT / "results" / "checkpoints")
    p.add_argument("--output", type=Path,
                   default=ROOT / "results" / "schemeA_campaign.json")
    return p


if __name__ == "__main__":
    a = parser().parse_args()
    run(a)
    print(a.output)
