#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / "core")]

from adaptive_search import evaluate, generate_complete_path
from rq1_common import (
    individual_dict,
    load_replay_context,
    make_initial_population,
    nondominated,
    save_json_gz,
)


def _params(args):
    return {
        k: (str(v) if isinstance(v, Path) else v)
        for k, v in vars(args).items()
        if k not in {"model", "checkpoint", "output"}
    }


def run(args):
    model, runtime, archive, checkpoint = load_replay_context(args.model, args.checkpoint)
    rng, pop = make_initial_population(
        runtime,
        archive,
        seed=args.seed,
        pop_size=args.pop_size,
        max_path_len=args.max_path_len,
        path_gen_retries=args.path_gen_retries,
        terminal_template_weight=args.terminal_template_weight,
        k_neighbors=args.k_neighbors,
        alpha=args.alpha,
        beta=args.beta,
    )

    evaluations = len(pop)
    pool = nondominated(pop)
    if len(pool) > args.archive_size:
        pool = rng.sample(pool, args.archive_size)
    history = []
    batch = 0

    def snapshot():
        history.append(
            {
                "batch": batch,
                "fitness_evaluations": evaluations,
                "pareto_front": [individual_dict(x) for x in pool],
            }
        )

    snapshot()
    t0 = time.perf_counter()

    while evaluations < args.fitness_evaluations:
        batch += 1
        n = min(args.pop_size, args.fitness_evaluations - evaluations)
        new = []
        for _ in range(n):
            ind = generate_complete_path(
                runtime,
                rng,
                args.max_path_len,
                args.terminal_template_weight,
                args.path_gen_retries,
            )
            evaluate(ind, runtime, archive, args.k_neighbors, args.alpha, args.beta)
            new.append(ind)
        evaluations += len(new)
        pool = nondominated(pool + new)
        if len(pool) > args.archive_size:
            pool = rng.sample(pool, args.archive_size)
        if batch % args.snapshot_interval == 0 or evaluations >= args.fitness_evaluations:
            snapshot()

    elapsed = time.perf_counter() - t0
    out = {
        "schema": "rq2-search-run-v2",
        "algorithm": "ObjectiveGuidedRandom",
        "model_id": model.get("model_id"),
        "checkpoint": str(args.checkpoint),
        "checkpoint_campaign_id": checkpoint.get("campaign_id"),
        "checkpoint_test_index": checkpoint.get("test_index"),
        "seed": args.seed,
        "parameters": _params(args),
        "fitness_evaluations": evaluations,
        "runtime_sec": elapsed,
        "final_pareto_front": [individual_dict(x) for x in pool],
        "pareto_front_history": history,
    }
    save_json_gz(args.output, out)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model", type=Path)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--pop-size", type=int, default=100)
    p.add_argument("--archive-size", type=int, default=100)
    p.add_argument("--fitness-evaluations", type=int, default=20100)
    p.add_argument("--max-path-len", type=int, default=30)
    p.add_argument("--path-gen-retries", type=int, default=30)
    p.add_argument("--terminal-template-weight", type=float, default=0.20)
    p.add_argument("--k-neighbors", type=int, default=5)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--snapshot-interval", type=int, default=10)
    args = p.parse_args()
    run(args)
    print(args.output)


if __name__ == "__main__":
    main()
