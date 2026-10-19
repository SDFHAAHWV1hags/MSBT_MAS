#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / "core"), str(ROOT / "brs_v0_4"), str(ROOT)]

from adaptive_search import (
    CampaignArchive,
    ModelRuntime,
    evaluate,
    generate_complete_path,
    update_archive,
)
from rq2_common import nondominated
from checkpoint_io import (
    CheckpointRecorder,
    decode_random_state,
    deserialize_archive,
    load_checkpoint,
    sha256_json,
)
from execution_adapter import (
    ModuleExecutor, MockExecutor, extract_llm_usage,
)
from file_provision import apply_hint, resolve_gaia_root
from brs_bridge import make_concretizer
from brs_grounding_policy import apply_leave_current_query_out
from brs_user_task_generator import GenerationFailure
from campaign_runner import (
    dependency_for_instance,
    provision_v2_attachments,
    rank_pareto_by_ideal,
)
from llm_client import load_llm

_active_obj_indices = tuple(
    getattr(sys.modules.get("adaptive_search"), "ACTIVE_OBJECTIVE_INDICES", (0, 1, 2, 3))
)
IS_THREE_OBJ = _active_obj_indices == (0, 1, 3)
METHOD_LABEL = (
    "ConstrainedRandom-3obj-BRS-v0.4" if IS_THREE_OBJ else "ConstrainedRandom-BRS-v0.4"
)
OBJECTIVE_GUIDANCE_LABEL = (
    "schemeA_3objectives_no_evolution"
    if IS_THREE_OBJ
    else "same_4_objectives_as_full_no_evolution"
)


NOT_EVALUATED = {
    "verdict": "NOT_EVALUATED",
    "reason": "Oracle construction is outside the current BRS design.",
}


def _prune_and_cap(front, cap, trunc_rng):
    front = nondominated(front)
    if cap is not None and len(front) > cap:
        trunc_rng.shuffle(front)
        front = front[:cap]
    return front


def random_front(runtime, archive, rng, trunc_rng, a):
    cap = a.pareto_archive_cap or None
    front = []
    pending = 0
    for _ in range(a.fitness_evaluations):
        x = generate_complete_path(
            runtime,
            rng,
            a.max_path_len,
            a.terminal_template_weight,
            a.path_gen_retries,
        )
        evaluate(x, runtime, archive, a.k_neighbors, a.alpha, a.beta)
        front.append(x)
        pending += 1
        if pending >= 100:
            front = _prune_and_cap(front, cap, trunc_rng)
            pending = 0
    return _prune_and_cap(front, cap, trunc_rng)


def select_and_concretize(
    runtime,
    archive,
    rng,
    concretizer,
    model,
    a,
):
    attempt_log = []
    total_search_sec = 0.0
    trunc_rng = random.Random((a.seed + 1) * 1000003 ^ 0x9E3779B9)

    for search_attempt in range(1, a.max_search_regenerations + 1):
        s0 = time.perf_counter()
        pareto = random_front(runtime, archive, rng, trunc_rng, a)
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
            return (
                pareto,
                candidate,
                testcase,
                total_search_sec,
                nlp_sec,
                attempt_log,
            )

    raise RuntimeError(
        "No concretizable target path was found after "
        f"{a.max_search_regenerations} random-front regenerations "
        f"({a.fitness_evaluations} draws each)."
    )


def run(a) -> dict:
    model = json.loads(a.model.read_text(encoding="utf-8"))
    runtime = ModelRuntime(model)
    archive = CampaignArchive()
    rng = random.Random(a.seed)

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

    executor = MockExecutor() if a.executor is None else ModuleExecutor(a.executor)

    cid = (
        a.campaign_id
        or f'{model.get("model_id")}_random_brs04_{a.llm_id}_seed{a.seed}'
    )

    params = {
        "fitness_evaluations": a.fitness_evaluations,
        "max_path_len": a.max_path_len,
        "path_gen_retries": a.path_gen_retries,
        "terminal_template_weight": a.terminal_template_weight,
        "k_neighbors": a.k_neighbors,
        "alpha": a.alpha,
        "beta": a.beta,
        "max_search_regenerations": a.max_search_regenerations,
        "pareto_archive_cap": a.pareto_archive_cap,
        "objective_guidance": OBJECTIVE_GUIDANCE_LABEL,
        "active_objective_indices": list(_active_obj_indices),
        "brs_library": str(a.brs_library),
        "grounding_mode": a.grounding_mode,
        "signature_policy": a.signature_policy,
    }

    recorder = CheckpointRecorder(
        a.checkpoint_root,
        cid,
        model,
        metadata={
            "method": METHOD_LABEL,
            "brs_version": "0.4",
            "llm_id": a.llm_id,
            "seed": a.seed,
            "fitness_evaluations": a.fitness_evaluations,
            "excluded_grounded_instances": excluded,
            "brs_library": str(a.brs_library),
            "signature_policy": a.signature_policy,
        },
    )

    if a.resume:
        latest = recorder.root / "LATEST"
        if not latest.exists():
            raise SystemExit(f"--resume: no checkpoint found under {recorder.root}")
        ck = load_checkpoint(recorder.root / latest.read_text(encoding="utf-8").strip())
        if ck.get("model_hash") != sha256_json(model):
            raise SystemExit(
                "--resume: model file hash differs from the checkpointed campaign"
            )
        ck_params = ck.get("search_parameters") or {}
        old_cap = ck_params.get("pareto_archive_cap", "<missing>")
        if old_cap != params.get("pareto_archive_cap"):
            raise SystemExit(
                "--resume: pareto_archive_cap differs from checkpoint "
                f"(checkpoint={old_cap!r}, current build="
                f"{params.get('pareto_archive_cap')!r}). Delete the old "
                "checkpoint directory or rerun with FRESH=1 under a new "
                "RUN_TAG; rounds produced under different search semantics "
                "must not share a campaign."
            )
        start_t = int(ck["test_index"]) + 1
        archive = deserialize_archive(ck["archive"])
        if ck.get("random_state_b64"):
            rng.setstate(decode_random_state(ck["random_state_b64"]))
        else:
            rng = random.Random(a.seed + start_t)
            print(
                "[resume] WARNING: checkpoint lacks rng state; reseeded (not bit-exact)",
                file=sys.stderr,
            )
        tests = []
        for k in range(1, start_t):
            row = load_checkpoint(recorder.root / f"C{k:03d}.json").get("last_round")
            if not row:
                raise SystemExit(
                    f"--resume: missing completed round C{k:03d}.json; "
                    "cannot reconstruct prior rows"
                )
            tests.append(row)
        print(
            f"[resume] restored {start_t - 1} completed round(s); "
            f"features={len(archive.covered_features)} "
            f"executed_tests={len(archive.executed_tests)}; "
            f"continuing at test {start_t}",
            file=sys.stderr,
        )
    else:
        recorder.save(
            0,
            archive,
            search_parameters=params,
            random_state=rng.getstate(),
        )
        start_t = 1
        tests = []

    provision_dir = Path(a.provision_dir)
    gaia_root_resolved = None
    seed_map = {}
    if a.provision_files:
        seed_map, gaia_root_resolved = provision_v2_attachments(
            concretizer.repo.instances, a.gaia_root, provision_dir, a.sandbox_root
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
                f"[provision] {len(seed_map)} attachment(s) copied into "
                f"{provision_dir.resolve()} from {gaia_root_resolved}",
                file=sys.stderr,
            )

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
                concretizer,
                model,
                a,
            )
        except RuntimeError as exc:
            search_sec = time.perf_counter() - s_search
            print(
                f"[random-brs04] test {t}/{a.num_tests} {exc} "
                "-> outcome=SEARCH_UNCONCRETIZABLE",
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
                    "reason": "No concretizable path found within the random "
                              "front regeneration budget.",
                },
                "timings": {},
                "mas_llm_usage": None,
                "concretization_attempts": [],
                "search_error": str(exc),
            }
            tests.append(row)
            recorder.save(
                t,
                archive,
                last_round=row,
                search_parameters=params,
                random_state=rng.getstate(),
            )
            continue

        dep = dependency_for_instance(tc, seed_map)
        query_sent = apply_hint(tc.query, dep)

        execution_result = executor.execute(
            query_sent,
            model,
            {
                "method": METHOD_LABEL,
                "test_index": t,
                "seed": a.seed,
                "model_path": str(a.model.resolve()),
                "target_path": selected.genes,
                "work_dir": str(provision_dir.resolve()) if a.provision_files else None,
                "file_dependency": dep,
            },
        )
        actual_path = executor.analyze_path(execution_result.raw_log, model)

        update_archive(
            archive,
            runtime,
            selected.genes,
            actual_path,
            query_sent,
            execution_result.outcome,
            execution_result.raw_log,
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
            "actual_path": [g.to_dict() for g in actual_path],
            "outcome": execution_result.outcome,
            "failure_signature": execution_result.failure_signature,
            "final_answer": execution_result.final_answer,
            "oracle_verdict": NOT_EVALUATED,
            "timings": execution_result.timings or {},
            "mas_llm_usage": extract_llm_usage(execution_result.raw_log),
            "concretization_attempts": concretization_attempts,
        }
        tests.append(row)

        recorder.save(
            t,
            archive,
            last_round=row,
            search_parameters=params,
            random_state=rng.getstate(),
        )
        print(
            f"[random-brs04] test {t}/{a.num_tests} "
            f"front={len(pf)} path_len={len(selected.genes)} "
            f"actual_len={len(actual_path)} outcome={execution_result.outcome}",
            file=sys.stderr,
        )

    output = {
        "method": METHOD_LABEL,
        "brs_version": "0.4",
        "model_id": model.get("model_id"),
        "llm_id": a.llm_id,
        "seed": a.seed,
        "num_tests": a.num_tests,
        "fitness_evaluations": a.fitness_evaluations,
        "excluded_grounded_instances": excluded,
        "parameters": params,
        "file_provisioning": {
            "enabled": bool(a.provision_files and gaia_root_resolved),
            "gaia_root": str(gaia_root_resolved) if gaia_root_resolved else None,
            "provision_dir": str(provision_dir.resolve()),
            "sandbox_root": a.sandbox_root or str(provision_dir.resolve()),
            "seed_files": len(seed_map),
        },
        "tests": tests,
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return output


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "RQ1 Constrained-Random baseline on BRS v0.4: equal-budget random "
            "path sampling with the same four objectives as Full, BRS v0.4 "
            "concretization, no NSGA-II evolution."
        )
    )
    p.add_argument("model", type=Path)
    p.add_argument("--llm-config", type=Path, required=True)
    p.add_argument("--llm-id", required=True)
    p.add_argument("--executor", type=Path, default=None)
    p.add_argument("--allow-mock", action="store_true")
    p.add_argument("--num-tests", type=int, default=30)
    p.add_argument("--seed", type=int, default=1)

    p.add_argument("--fitness-evaluations", type=int, default=20100)
    p.add_argument("--max-path-len", type=int, default=30)
    p.add_argument("--path-gen-retries", type=int, default=30)
    p.add_argument("--terminal-template-weight", type=float, default=0.2)
    p.add_argument("--k-neighbors", type=int, default=5)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument(
        "--max-search-regenerations",
        type=int,
        default=3,
        help="complete random-front regenerations when no candidate is BRS-v0.4 concretizable",
    )
    p.add_argument(
        "--pareto-archive-cap",
        type=int,
        default=100,
        help=(
            "max non-dominated archive size after every 100-draw batch; "
            "oversize archives are uniformly randomly truncated with a "
            "dedicated RNG (aligned with Full's NSGA-II population 100); "
            "0 disables truncation"
        ),
    )

    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--max-tokens", type=int, default=16000)
    p.add_argument("--candidates", type=int, default=3)
    p.add_argument("--grounding-mode", choices=["bootstrap", "strict"],
                   default="bootstrap")
    p.add_argument("--signature-policy", choices=["exact", "collapse_adjacent"],
                   default="exact")
    p.add_argument("--allow-source-task-fallback", action="store_true")
    p.add_argument(
        "--brs-library",
        type=Path,
        default=ROOT / "brs_v0_4" / "library",
    )

    p.add_argument("--campaign-id", default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument(
        "--checkpoint-root",
        type=Path,
        default=ROOT / "results" / "checkpoints",
    )

    p.add_argument("--gaia-root", type=Path, default=None)
    p.add_argument(
        "--provision-dir",
        type=Path,
        default=ROOT / "results" / "provisioned_files_brs_v04",
    )
    p.add_argument("--sandbox-root", default=None)
    p.add_argument(
        "--no-provision-files",
        dest="provision_files",
        action="store_false",
    )
    p.set_defaults(provision_files=True)

    p.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "rq1_random_brs04.json",
    )
    a = p.parse_args()
    run(a)
    print(a.output)


if __name__ == "__main__":
    main()
