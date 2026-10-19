#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / "core")]

from adaptive_search import (
    Individual,
    OBJECTIVE_DIRECTIONS,
    crossover_complete_paths,
    evaluate,
    mutate_complete_path,
)
from rq1_common import (
    individual_dict,
    load_replay_context,
    make_initial_population,
    nondominated,
    save_json_gz,
)


def dominates(a: Individual, b: Individual) -> bool:
    strict = False
    for va, vb, direction in zip(a.objectives, b.objectives, OBJECTIVE_DIRECTIONS):
        aa, bb = direction * va, direction * vb
        if aa < bb:
            return False
        if aa > bb:
            strict = True
    return strict


def normalized_vectors(pop: Sequence[Individual]):
    if not pop:
        return []
    mins = [min(x.objectives[j] for x in pop) for j in range(4)]
    maxs = [max(x.objectives[j] for x in pop) for j in range(4)]
    out = []
    for x in pop:
        v = []
        for j, z in enumerate(x.objectives):
            lo, hi = mins[j], maxs[j]
            v.append(0.0 if hi <= lo else (z - lo) / (hi - lo))
        out.append(v)
    return out


def spea2_fitness(union: Sequence[Individual]):
    n = len(union)
    strength = [
        sum(1 for j, b in enumerate(union) if i != j and dominates(a, b))
        for i, a in enumerate(union)
    ]
    raw = [
        sum(strength[j] for j, b in enumerate(union) if i != j and dominates(b, a))
        for i, a in enumerate(union)
    ]

    vec = normalized_vectors(union)
    k = max(1, int(math.sqrt(n)))
    fitness = []
    for i in range(n):
        distances = sorted(math.dist(vec[i], vec[j]) for j in range(n) if j != i)
        sigma = distances[min(k - 1, len(distances) - 1)] if distances else 0.0
        density = 1.0 / (sigma + 2.0)
        fitness.append(raw[i] + density)
    return fitness, vec


def truncate_archive(indices: List[int], vec: List[List[float]], target: int) -> List[int]:
    indices = list(indices)
    while len(indices) > target:
        signatures = []
        for i in indices:
            distances = sorted(math.dist(vec[i], vec[j]) for j in indices if j != i)
            signatures.append((tuple(distances), i))
        _, remove = min(signatures, key=lambda x: (x[0], x[1]))
        indices.remove(remove)
    return indices


def environmental(union: Sequence[Individual], archive_size: int):
    fitness, vec = spea2_fitness(union)
    chosen = [i for i, f in enumerate(fitness) if f < 1.0]
    if len(chosen) < archive_size:
        rest = sorted((fitness[i], i) for i in range(len(union)) if i not in chosen)
        chosen.extend(i for _, i in rest[: archive_size - len(chosen)])
    elif len(chosen) > archive_size:
        chosen = truncate_archive(chosen, vec, archive_size)

    archive = [union[i].clone() for i in chosen]
    lookup = {
        (tuple(x.objectives), tuple(g.key for g in x.genes)): fitness[i]
        for i, x in enumerate(union)
    }
    return archive, lookup


def tournament(archive: Sequence[Individual], lookup: Dict, rng: random.Random) -> Individual:
    a, b = rng.choice(archive), rng.choice(archive)
    ka = (tuple(a.objectives), tuple(g.key for g in a.genes))
    kb = (tuple(b.objectives), tuple(g.key for g in b.genes))
    fa, fb = lookup.get(ka, float("inf")), lookup.get(kb, float("inf"))
    if fa < fb:
        return a
    if fb < fa:
        return b
    return a if rng.random() < 0.5 else b


def _params(args):
    return {
        k: (str(v) if isinstance(v, Path) else v)
        for k, v in vars(args).items()
        if k not in {"model", "checkpoint", "output"}
    }


def run(args):
    model, runtime, campaign, checkpoint = load_replay_context(args.model, args.checkpoint)
    rng, pop = make_initial_population(
        runtime,
        campaign,
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
    history = []
    archive, lookup = environmental(pop, args.archive_size)

    def snapshot(generation: int):
        front = nondominated(archive)
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
        children = []
        while len(children) < args.pop_size:
            p1 = tournament(archive, lookup, rng)
            p2 = tournament(archive, lookup, rng)

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
            children.extend([c1, c2])

        pop = children[: args.pop_size]
        for ind in pop:
            evaluate(ind, runtime, campaign, args.k_neighbors, args.alpha, args.beta)
        evaluations += len(pop)

        archive, lookup = environmental(pop + archive, args.archive_size)
        if generation % args.snapshot_interval == 0 or generation == args.generations:
            snapshot(generation)

    elapsed = time.perf_counter() - t0
    front = nondominated(archive)
    out = {
        "schema": "rq2-search-run-v2",
        "algorithm": "SPEA2",
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
    p.add_argument("--archive-size", type=int, default=100)
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
