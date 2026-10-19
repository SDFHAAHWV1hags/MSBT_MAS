#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [
    str(ROOT / "core"),
    str(ROOT / "brs_v0_4"),
    str(ROOT),
]

from adaptive_search import (
    CampaignArchive,
    ModelRuntime,
    OBJECTIVE_DIRECTIONS,
    nsga2_complete_path_search,
    update_archive,
)
from checkpoint_io import (
    CheckpointRecorder,
    load_checkpoint,
    deserialize_archive,
    decode_random_state,
    sha256_json,
)
from execution_adapter import (
    ModuleExecutor, MockExecutor, extract_llm_usage,
)
from llm_client import load_llm
from brs_bridge import make_concretizer
from brs_grounding_policy import apply_leave_current_query_out
from brs_user_task_generator import GenerationFailure
from file_provision import FileIndex, resolve_gaia_root, sandbox_hint, apply_hint


def provision_v2_attachments(instances, gaia_root, provision_dir, sandbox_root=None):
    root = resolve_gaia_root(gaia_root)
    if root is None:
        return {}, None
    index = FileIndex()
    provision_dir = Path(provision_dir)
    provision_dir.mkdir(parents=True, exist_ok=True)
    visible = sandbox_root or str(provision_dir.resolve())
    seed_map = {}
    for inst in instances:
        sid = str(inst.get("source_id", "")).strip()
        if not sid or sid in seed_map:
            continue
        for hit in index.lookup(sid):
            host = root / hit["split"] / hit["file_name"]
            if host.exists():
                dst = provision_dir / hit["file_name"]
                if not dst.exists():
                    shutil.copy2(host, dst)
                seed_map[sid] = {
                    "file_name": hit["file_name"],
                    "split": hit["split"],
                    "host_path": str(host),
                    "sandbox_path": f"{visible.rstrip('/')}/{hit['file_name']}",
                }
                break
    return seed_map, root


def dependency_for_instance(testcase, seed_map):
    inst = testcase.grounded_instance or {}
    sid = str(inst.get("source_id") or "").strip()
    info = seed_map.get(sid)
    if not info:
        return None
    return {
        "question_id": sid,
        "file_name": info["file_name"],
        "split": info.get("split", ""),
        "host_path": info["host_path"],
        "sandbox_path": info["sandbox_path"],
        "hint": sandbox_hint(info),
    }


def rank_pareto_by_ideal(pareto):
    if not pareto:
        return []
    vals = [x.objectives for x in pareto]
    mins = [min(v[j] for v in vals) for j in range(4)]
    maxs = [max(v[j] for v in vals) for j in range(4)]

    def distance(x):
        u = []
        for j, (v, direction) in enumerate(zip(x.objectives, OBJECTIVE_DIRECTIONS)):
            lo, hi = mins[j], maxs[j]
            if hi <= lo:
                z = 1.0
            elif direction > 0:
                z = (v - lo) / (hi - lo)
            else:
                z = (hi - v) / (hi - lo)
            u.append(z)
        return math.sqrt(sum((1.0 - z) ** 2 for z in u))

    return sorted(
        [x.clone() for x in pareto],
        key=lambda x: (distance(x), len(x.genes), tuple(g.key for g in x.genes)),
    )


def select_and_concretize(
    runtime,
    archive,
    rng,
    concretizer,
    search_params,
    model,
    max_search_regenerations,
):
    attempt_log = []
    total_search_sec = 0.0

    for search_attempt in range(1, max_search_regenerations + 1):
        s0 = time.perf_counter()
        pareto, _ = nsga2_complete_path_search(
            runtime, archive, rng, **search_params
        )
        total_search_sec += time.perf_counter() - s0

        ranked = rank_pareto_by_ideal(pareto)
        for rank, candidate in enumerate(ranked, 1):
            availability = concretizer.can_concretize(candidate.genes, model)
            if not availability.get("ok"):
                attempt_log.append(
                    {
                        "search_attempt": search_attempt,
                        "pareto_rank": rank,
                        "path": [g.to_dict() for g in candidate.genes],
                        "availability": availability,
                    }
                )
                continue

            g0 = time.perf_counter()
            try:
                testcase = concretizer.generate(candidate.genes, model, rng)
            except GenerationFailure as e:
                attempt_log.append(
                    {
                        "search_attempt": search_attempt,
                        "pareto_rank": rank,
                        "path": [g.to_dict() for g in candidate.genes],
                        "availability": availability,
                        "generation_failure": {
                            "reason": e.reason,
                            "details": e.details,
                        },
                    }
                )
                continue
            nlp_sec = time.perf_counter() - g0
            return pareto, candidate, testcase, total_search_sec, nlp_sec, attempt_log

    raise RuntimeError(
        "No concretizable target path was found after "
        f"{max_search_regenerations} complete search regenerations."
    )


def run(a):
    model = json.loads(a.model.read_text(encoding="utf-8"))
    runtime = ModelRuntime(model)
    rng = random.Random(a.seed)
    archive = CampaignArchive()

    chat = load_llm(a.llm_config, a.llm_id, allow_mock=a.allow_mock)
    con = make_concretizer(
        chat,
        a.temperature,
        a.max_tokens,
        library_root=a.brs_library,
        grounding_mode=a.grounding_mode,
        signature_policy=a.signature_policy,
        n_surface_candidates=a.candidates,
        allow_source_task_fallback=a.allow_source_task_fallback,
    )
    excluded = apply_leave_current_query_out(con, model)

    provision_dir = Path(a.provision_dir)
    seed_map = {}
    gaia_root_resolved = None
    if a.provision_files:
        seed_map, gaia_root_resolved = provision_v2_attachments(
            con.repo.instances, a.gaia_root, provision_dir, a.sandbox_root
        )
        con.set_attachment_map(seed_map)
        if gaia_root_resolved is None:
            print(
                "[provision] WARNING: GAIA 2023 root not found; file-dependent "
                "tasks will run without attachments (use --gaia-root or set "
                "MAS_GAIA_2023).",
                file=sys.stderr,
            )
        else:
            print(
                f"[provision] {len(seed_map)} attachment(s) copied into "
                f"{provision_dir.resolve()} from {gaia_root_resolved}",
                file=sys.stderr,
            )

    exe = MockExecutor() if a.executor is None else ModuleExecutor(a.executor)
    cid = a.campaign_id or f"{model.get('model_id')}_full_brs04_{a.llm_id}_seed{a.seed}"

    params = {
        "pop_size": a.pop_size,
        "generations": a.generations,
        "crossover_rate": a.crossover_rate,
        "mutation_rate": a.mutation_rate,
        "max_path_len": a.max_path_len,
        "path_gen_retries": a.path_gen_retries,
        "terminal_template_weight": a.terminal_template_weight,
        "k_neighbors": a.k_neighbors,
        "alpha": a.alpha,
        "beta": a.beta,
    }

    rec = CheckpointRecorder(
        a.checkpoint_root,
        cid,
        model,
        metadata={
            "method": "Full-BRS-v0.4",
            "brs_version": "0.4",
            "llm_id": a.llm_id,
            "seed": a.seed,
            "excluded_grounded_instances": excluded,
            "brs_library": str(a.brs_library),
            "grounding_mode": a.grounding_mode,
            "signature_policy": a.signature_policy,
            "provision_files": bool(a.provision_files and gaia_root_resolved),
            "provision_dir": str(provision_dir.resolve()),
            "seed_file_count": len(seed_map),
        },
    )
    if getattr(a, "resume", False):
        latest = rec.root / "LATEST"
        if not latest.exists():
            raise SystemExit(f"--resume: no checkpoint found under {rec.root}")
        ck = load_checkpoint(rec.root / latest.read_text(encoding="utf-8").strip())
        if ck.get("model_hash") != sha256_json(model):
            raise SystemExit("--resume: model file hash differs from the checkpointed campaign")
        start_t = int(ck["test_index"]) + 1
        archive = deserialize_archive(ck["archive"])
        rng.setstate(decode_random_state(ck["random_state_b64"]))
        tests = []
        for k in range(1, start_t):
            row = load_checkpoint(rec.root / f"C{k:03d}.json").get("last_round")
            if not row:
                raise SystemExit(f"--resume: missing completed round C{k:03d}.json")
            tests.append(row)
        print(
            f"[resume] restored {start_t - 1} completed round(s); continuing at test {start_t}",
            file=sys.stderr,
        )
    else:
        rec.save(0, archive, last_round={}, search_parameters=params, random_state=rng.getstate())
        start_t = 1
        tests = []

    for t in range(start_t, a.num_tests + 1):
        s_search = time.perf_counter()
        try:
            (
                pf,
                selected,
                tc,
                search_sec,
                nlp_sec,
                concretization_attempts,
            ) = select_and_concretize(
                runtime,
                archive,
                rng,
                con,
                params,
                model,
                a.max_search_regenerations,
            )
        except RuntimeError as exc:
            search_sec = time.perf_counter() - s_search
            print(
                f"[full-brs04] test {t}/{a.num_tests} {exc} -> outcome=SEARCH_UNCONCRETIZABLE",
                file=sys.stderr,
            )
            row = {
                "test_index": t,
                "search_sec": search_sec,
                "selected_path": None,
                "selected_objectives": None,
                "testcase": None,
                "query_sent": None,
                "file_dependency": None,
                "nlp_generation_sec": 0.0,
                "actual_path": [],
                "outcome": "SEARCH_UNCONCRETIZABLE",
                "failure_signature": "search_unconcretizable",
                "final_answer": None,
                "oracle_verdict": {
                    "verdict": "NOT_EVALUATED",
                    "reason": "No concretizable path found within the search regeneration budget.",
                },
                "timings": {},
                "mas_llm_usage": None,
                "concretization_attempts": [],
                "search_error": str(exc),
            }
            tests.append(row)
            rec.save(
                t,
                archive,
                last_round=row,
                search_parameters=params,
                random_state=rng.getstate(),
            )
            continue

        dep = dependency_for_instance(tc, seed_map)
        query_sent = apply_hint(tc.query, dep)
        er = exe.execute(
            query_sent,
            model,
            {
                "method": "Full-BRS-v0.4",
                "test_index": t,
                "model_path": str(a.model.resolve()),
                "target_path": selected.genes,
                "work_dir": str(provision_dir.resolve()) if a.provision_files else None,
                "file_dependency": dep,
            },
        )
        actual = exe.analyze_path(er.raw_log, model)

        update_archive(
            archive,
            runtime,
            selected.genes,
            actual,
            query_sent,
            er.outcome,
            er.raw_log,
        )

        row = {
            "test_index": t,
            "search_sec": search_sec,
            "pareto_front": [
                {
                    "path": [g.to_dict() for g in x.genes],
                    "objectives": list(x.objectives),
                }
                for x in pf
            ],
            "selected_path": [g.to_dict() for g in selected.genes],
            "selected_objectives": list(selected.objectives),
            "testcase": asdict(tc),
            "query_sent": query_sent,
            "file_dependency": dep,
            "nlp_generation_sec": nlp_sec,
            "actual_path": [g.to_dict() for g in actual],
            "outcome": er.outcome,
            "failure_signature": er.failure_signature,
            "final_answer": er.final_answer,
            "oracle_verdict": {
                "verdict": "NOT_EVALUATED",
                "reason": "Oracle construction is outside the current BRS design.",
            },
            "timings": er.timings or {},
            "mas_llm_usage": extract_llm_usage(er.raw_log),
            "concretization_attempts": concretization_attempts,
        }
        tests.append(row)
        rec.save(
            t,
            archive,
            last_round=row,
            search_parameters=params,
            random_state=rng.getstate(),
        )

    out = {
        "method": "Full-BRS-v0.4",
        "brs_version": "0.4",
        "model_id": model.get("model_id"),
        "llm_id": a.llm_id,
        "seed": a.seed,
        "excluded_grounded_instances": excluded,
        "file_provisioning": {
            "enabled": bool(a.provision_files and gaia_root_resolved),
            "gaia_root": str(gaia_root_resolved) if gaia_root_resolved else None,
            "provision_dir": str(provision_dir.resolve()),
            "sandbox_root": a.sandbox_root or str(provision_dir.resolve()),
            "seed_files": len(seed_map),
        },
        "parameters": params,
        "tests": tests,
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def parser():
    p = argparse.ArgumentParser()
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
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--temperature", type=float, default=.2)
    p.add_argument("--max-tokens", type=int, default=16000)
    p.add_argument("--candidates", type=int, default=3)
    p.add_argument("--grounding-mode", choices=["bootstrap", "strict"], default="bootstrap")
    p.add_argument(
        "--signature-policy",
        choices=["exact", "collapse_adjacent"],
        default="exact",
        help="v0.4 canonical policy is exact (elements are reusable, so "
             "adjacent equal operations are not collapsed)",
    )
    p.add_argument("--allow-source-task-fallback", action="store_true")
    p.add_argument("--max-search-regenerations", type=int, default=3)
    p.add_argument(
        "--brs-library",
        type=Path,
        default=ROOT / "brs_v0_4" / "library",
        help="v0.4 fragment library + carried bootstrap grounding corpus",
    )
    p.add_argument("--campaign-id", default=None)
    p.add_argument(
        "--resume",
        action="store_true",
        help="restore archive+rng from the LATEST checkpoint and continue at the next test",
    )
    p.add_argument(
        "--checkpoint-root",
        type=Path,
        default=ROOT / "results" / "checkpoints",
    )
    p.add_argument(
        "--gaia-root",
        type=Path,
        default=None,
        help="GAIA 2023 root (validation/test with attachments); "
             "default: $MAS_GAIA_2023 or auto-detect",
    )
    p.add_argument(
        "--provision-dir",
        type=Path,
        default=ROOT / "results" / "provisioned_files_brs_v04",
        help="shared sandbox directory attachments are copied into and the MAS runs in",
    )
    p.add_argument(
        "--sandbox-root",
        default=None,
        help="file path as seen from inside the MAS (/workspace for Docker); "
             "default: host-absolute provision dir",
    )
    p.add_argument(
        "--no-provision-files",
        dest="provision_files",
        action="store_false",
        help="disable GAIA attachment provisioning (file-dependent tasks run without files)",
    )
    p.set_defaults(provision_files=True)
    p.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "full_brs_v04_campaign.json",
    )
    return p


if __name__ == "__main__":
    a = parser().parse_args()
    run(a)
    print(a.output)
