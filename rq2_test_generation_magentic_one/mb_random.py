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
    generate_complete_path,
    update_archive,
)
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
)
from llm_client import load_llm


PURE_RANDOM_TERMINAL_TEMPLATE_WEIGHT = 1.0

NOT_EVALUATED = {
    "verdict": "NOT_EVALUATED",
    "reason": "Oracle construction is outside the current BRS design.",
}


def generate_model_random_path(runtime, rng, *, max_path_len, path_gen_retries):
    return generate_complete_path(
        runtime,
        rng,
        max_path_len,
        PURE_RANDOM_TERMINAL_TEMPLATE_WEIGHT,
        path_gen_retries,
    )


def draw_concretizable(runtime, concretizer, model, rng, args):
    draw_log = []
    t_start = time.perf_counter()
    for draw in range(1, args.max_concretizable_draws + 1):
        selected = generate_model_random_path(
            runtime,
            rng,
            max_path_len=args.max_path_len,
            path_gen_retries=args.path_gen_retries,
        )
        ok, end_state, _ = runtime.simulate(selected.genes)
        if (
            not ok
            or not runtime.is_terminal(end_state)
            or not selected.genes
        ):
            draw_log.append({"draw": draw, "rejected": "invalid_model_path"})
            continue
        availability = concretizer.can_concretize(selected.genes, model)
        if not availability.get("ok"):
            draw_log.append(
                {
                    "draw": draw,
                    "path_len": len(selected.genes),
                    "rejected": availability.get("reason"),
                    "operation_signature": availability.get(
                        "operation_signature"
                    ),
                }
            )
            continue
        return selected, draw_log, time.perf_counter() - t_start
    raise RuntimeError(
        "MB-Random: no concretizable model-valid path found after "
        f"{args.max_concretizable_draws} draws"
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
        or f'{model.get("model_id")}_mb_random_brs04_{a.llm_id}_seed{a.seed}'
    )

    path_parameters = {
        "max_path_len": a.max_path_len,
        "path_gen_retries": a.path_gen_retries,
        "terminal_template_weight": PURE_RANDOM_TERMINAL_TEMPLATE_WEIGHT,
        "objective_guidance": False,
        "candidate_pool_size": 1,
        "brs_library": str(a.brs_library),
        "grounding_mode": a.grounding_mode,
        "signature_policy": a.signature_policy,
        "max_concretizable_draws": a.max_concretizable_draws,
    }

    recorder = CheckpointRecorder(
        a.checkpoint_root,
        cid,
        model,
        metadata={
            "method": "MB-Random-BRS-v0.4",
            "brs_version": "0.4",
            "llm_id": a.llm_id,
            "seed": a.seed,
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
            search_parameters=path_parameters,
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
        selected, draw_log, path_generation_sec = draw_concretizable(
            runtime, concretizer, model, rng, a
        )

        g0 = time.perf_counter()
        gen_err = None
        testcase = None
        for attempt in range(1, a.nlp_retries + 1):
            try:
                testcase = concretizer.generate(selected.genes, model, rng)
                gen_err = None
                break
            except GenerationFailure as exc:
                gen_err = f"{exc.reason}: {exc.details}"
                if attempt < a.nlp_retries:
                    selected, more_log, extra_sec = draw_concretizable(
                        runtime, concretizer, model, rng, a
                    )
                    draw_log.extend(more_log)
                    path_generation_sec += extra_sec
        nlp_generation_sec = time.perf_counter() - g0
        if gen_err is not None or testcase is None:
            raise RuntimeError(
                f"NLP concretization failed after {a.nlp_retries} attempts "
                f"(test {t}): {gen_err}"
            )
        nlp_attempts = attempt

        dep = dependency_for_instance(testcase, seed_map)
        query_sent = apply_hint(testcase.query, dep)

        execution_result = executor.execute(
            query_sent,
            model,
            {
                "method": "MB-Random-BRS-v0.4",
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
            "search_sec": path_generation_sec,
            "path_generation_sec": path_generation_sec,
            "selection_mode": "model_valid_random",
            "objective_guidance": False,
            "candidate_pool_size": 1,
            "pareto_front": [],
            "selected_path": [g.to_dict() for g in selected.genes],
            "selected_objectives": None,
            "testcase": asdict(testcase),
            "query_sent": query_sent,
            "file_dependency": dep,
            "nlp_generation_sec": nlp_generation_sec,
            "nlp_attempts": nlp_attempts,
            "path_draw_log": draw_log,
            "actual_path": [g.to_dict() for g in actual_path],
            "outcome": execution_result.outcome,
            "failure_signature": execution_result.failure_signature,
            "final_answer": execution_result.final_answer,
            "oracle_verdict": NOT_EVALUATED,
            "timings": execution_result.timings or {},
            "mas_llm_usage": extract_llm_usage(execution_result.raw_log),
        }
        tests.append(row)

        recorder.save(
            t,
            archive,
            last_round=row,
            search_parameters=path_parameters,
            random_state=rng.getstate(),
        )
        print(
            f"[mb-random-brs04] test {t}/{a.num_tests} draws={len(draw_log) + 1} "
            f"path_len={len(selected.genes)} actual_len={len(actual_path)} "
            f"outcome={execution_result.outcome}",
            file=sys.stderr,
        )

    output = {
        "method": "MB-Random-BRS-v0.4",
        "brs_version": "0.4",
        "model_id": model.get("model_id"),
        "llm_id": a.llm_id,
        "seed": a.seed,
        "num_tests": a.num_tests,
        "excluded_grounded_instances": excluded,
        "parameters": path_parameters,
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
            "RQ1 MB-Random baseline on BRS v0.4: random model-valid paths drawn "
            "until v0.4-concretizable, then BRS concretized and executed."
        )
    )
    p.add_argument("model", type=Path)
    p.add_argument("--llm-config", type=Path, required=True)
    p.add_argument("--llm-id", required=True)
    p.add_argument("--executor", type=Path, default=None)
    p.add_argument("--allow-mock", action="store_true")
    p.add_argument("--num-tests", type=int, default=30)
    p.add_argument("--seed", type=int, default=1)

    p.add_argument("--max-path-len", type=int, default=30)
    p.add_argument("--path-gen-retries", type=int, default=30)
    p.add_argument(
        "--max-concretizable-draws",
        type=int,
        default=30,
        help="random path draws per test while can_concretize rejects",
    )

    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--max-tokens", type=int, default=16000)
    p.add_argument("--candidates", type=int, default=3)
    p.add_argument("--grounding-mode", choices=["bootstrap", "strict"],
                   default="bootstrap")
    p.add_argument("--signature-policy", choices=["exact", "collapse_adjacent"],
                   default="exact")
    p.add_argument("--allow-source-task-fallback", action="store_true")
    p.add_argument("--nlp-retries", type=int, default=3)
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
        default=ROOT / "results" / "rq1_mb_random_brs04.json",
    )
    a = p.parse_args()
    run(a)
    print(a.output)


if __name__ == "__main__":
    main()
