#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / "core")]

from adaptive_search import (
    crossover_complete_paths,
    environmental_selection,
    evaluate,
    mutate_complete_path,
    rank_and_crowding,
    tournament,
)
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

    rank_and_crowding(pop)
    evaluations = len(pop)
    history = []

    def snapshot(generation: int):
        front = nondominated(pop)
        history.append(
            {
                "generation": generation,
                "fitness_evaluations": evaluations,
                "pareto_front": [individual_dict(x) for x in front],
            }
        )

    snapshot(0)
    t0 = time.perf_counter()

    for generation in range(1, args.generations + 1):
        offspring = []
        while len(offspring) < args.pop_size:
            p1 = tournament(rng.choice(pop), rng.choice(pop), rng)
            p2 = tournament(rng.choice(pop), rng.choice(pop), rng)

            if rng.random() < args.crossover_rate:
                c1, c2 = crossover_complete_paths(p1, p2, runtime, rng, args.max_path_len)
            else:
                c1, c2 = p1.clone(), p2.clone()

            if rng.random() < args.mutation_rate:
                c1 = mutate_complete_path(
                    c1,
                    runtime,
                    rng,
                    args.max_path_len,
                    args.terminal_template_weight,
                    args.path_gen_retries,
                )
            if rng.random() < args.mutation_rate:
                c2 = mutate_complete_path(
                    c2,
                    runtime,
                    rng,
                    args.max_path_len,
                    args.terminal_template_weight,
                    args.path_gen_retries,
                )
            offspring.extend([c1, c2])

        offspring = offspring[: args.pop_size]
        for ind in offspring:
            evaluate(ind, runtime, archive, args.k_neighbors, args.alpha, args.beta)
        evaluations += len(offspring)
        pop = environmental_selection(pop + offspring, args.pop_size)

        if generation % args.snapshot_interval == 0 or generation == args.generations:
            snapshot(generation)

    elapsed = time.perf_counter() - t0
    front = nondominated(pop)
    out = {
        "schema": "rq2-search-run-v2",
        "algorithm": "NSGA-II",
        "model_id": model.get("model_id"),
        "checkpoint": str(args.checkpoint),
        "checkpoint_campaign_id": checkpoint.get("campaign_id"),
        "checkpoint_test_index": checkpoint.get("test_index"),
        "seed": args.seed,
        "parameters": _params(args),
        "fitness_evaluations": evaluations,
        "runtime_sec": elapsed,
        "final_pareto_front": [individual_dict(x) for x in front],
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
    p.add_argument("--generations", type=int, default=200)
    p.add_argument("--crossover-rate", type=float, default=0.80)
    p.add_argument("--mutation-rate", type=float, default=0.35)
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
