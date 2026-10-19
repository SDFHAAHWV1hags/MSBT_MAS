#!/usr/bin/env python3
from __future__ import annotations

import gzip
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / "core")]

from adaptive_search import (
    Individual,
    ModelRuntime,
    OBJECTIVE_DIRECTIONS,
    evaluate,
    generate_complete_path,
)
from checkpoint_io import (
    load_checkpoint,
    restore_archive,
    sha256_json,
)


def load_json_any(path: Path) -> Dict[str, Any]:
    path = Path(path)
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    return json.loads(path.read_text(encoding="utf-8"))


def save_json_gz(path: Path, obj: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix != ".gz":
        path = path.with_suffix(path.suffix + ".gz") if path.suffix else path.with_suffix(".json.gz")
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_replay_context(model_path: Path, checkpoint_path: Path):
    model = json.loads(Path(model_path).read_text(encoding="utf-8"))
    checkpoint = load_checkpoint(Path(checkpoint_path))

    if checkpoint.get("model_id") != model.get("model_id"):
        raise ValueError(
            f"Model/checkpoint mismatch: model={model.get('model_id')!r}, "
            f"checkpoint={checkpoint.get('model_id')!r}"
        )

    expected_hash = checkpoint.get("model_hash")
    if expected_hash and expected_hash != sha256_json(model):
        raise ValueError("Checkpoint model hash does not match the supplied model file.")

    runtime = ModelRuntime(model)
    archive = restore_archive(checkpoint)
    return model, runtime, archive, checkpoint


def make_initial_population(
    runtime: ModelRuntime,
    archive,
    *,
    seed: int,
    pop_size: int,
    max_path_len: int,
    path_gen_retries: int,
    terminal_template_weight: float,
    k_neighbors: int,
    alpha: float,
    beta: float,
):
    rng = random.Random(seed)
    pop: List[Individual] = []
    for _ in range(pop_size):
        ind = generate_complete_path(
            runtime,
            rng,
            max_path_len,
            terminal_template_weight,
            path_gen_retries,
        )
        evaluate(ind, runtime, archive, k_neighbors, alpha, beta)
        pop.append(ind)
    return rng, pop


def _dominates_individual(a: Individual, b: Individual) -> bool:
    strict = False
    for va, vb, direction in zip(a.objectives, b.objectives, OBJECTIVE_DIRECTIONS):
        aa = direction * va
        bb = direction * vb
        if aa < bb:
            return False
        if aa > bb:
            strict = True
    return strict


def nondominated(pop: Sequence[Individual]) -> List[Individual]:
    out: List[Individual] = []
    for i, a in enumerate(pop):
        if not any(i != j and _dominates_individual(b, a) for j, b in enumerate(pop)):
            out.append(a.clone())
    return out


def individual_dict(ind: Individual) -> Dict[str, Any]:
    return {
        "path": [g.to_dict() for g in ind.genes],
        "path_length": len(ind.genes),
        "objectives": list(ind.objectives),
    }




def empirical_bounds(front_sets: Sequence[Sequence[Sequence[float]]]) -> Dict[str, List[float]]:
    points = [list(map(float, p)) for fs in front_sets for p in fs]
    if not points:
        raise ValueError("Cannot compute empirical bounds from zero points.")
    dim = len(points[0])
    if dim != len(OBJECTIVE_DIRECTIONS):
        raise ValueError(f"Expected {len(OBJECTIVE_DIRECTIONS)} objectives, got {dim}.")
    mins = [min(p[j] for p in points) for j in range(dim)]
    maxs = [max(p[j] for p in points) for j in range(dim)]
    return {"mins": mins, "maxs": maxs}


def normalize_points(
    points: Sequence[Sequence[float]],
    bounds: Dict[str, Sequence[float]],
) -> List[List[float]]:
    mins = list(map(float, bounds["mins"]))
    maxs = list(map(float, bounds["maxs"]))
    out: List[List[float]] = []
    for p in points:
        q: List[float] = []
        for j, (value, direction) in enumerate(zip(map(float, p), OBJECTIVE_DIRECTIONS)):
            lo, hi = mins[j], maxs[j]
            if hi <= lo:
                z = 0.0
            elif direction > 0:
                z = (hi - value) / (hi - lo)
            else:
                z = (value - lo) / (hi - lo)
            q.append(min(1.0, max(0.0, z)))
        out.append(q)
    return out


def _dominates_min(a: Sequence[float], b: Sequence[float]) -> bool:
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))


def nd_min(points: Sequence[Sequence[float]]) -> List[List[float]]:
    pts: List[List[float]] = []
    seen = set()
    for p in points:
        t = tuple(float(x) for x in p)
        if t not in seen:
            seen.add(t)
            pts.append(list(t))
    out: List[List[float]] = []
    for i, a in enumerate(pts):
        if not any(i != j and _dominates_min(b, a) for j, b in enumerate(pts)):
            out.append(a)
    return out


def _hv_union_origin(boxes: Sequence[Sequence[float]], dim: int) -> float:
    valid = [list(map(float, b[:dim])) for b in boxes if all(float(x) > 0 for x in b[:dim])]
    if not valid:
        return 0.0
    if dim == 1:
        return max(b[0] for b in valid)

    levels = sorted({b[dim - 1] for b in valid})
    volume = 0.0
    prev = 0.0
    for level in levels:
        width = level - prev
        if width > 0:
            active = [b[: dim - 1] for b in valid if b[dim - 1] >= level]
            volume += width * _hv_union_origin(active, dim - 1)
        prev = level
    return volume


def hypervolume_min(
    points: Sequence[Sequence[float]],
    reference: Sequence[float],
) -> float:
    ref = list(map(float, reference))
    pts = nd_min(points)
    boxes: List[List[float]] = []
    for p in pts:
        if len(p) != len(ref):
            raise ValueError("Point/reference dimensionality mismatch.")
        if any(float(x) > r for x, r in zip(p, ref)):
            continue
        q = [r - float(x) for x, r in zip(p, ref)]
        if all(x >= 0 for x in q):
            boxes.append(q)
    return _hv_union_origin(boxes, len(ref))


def igd_plus(
    approximation: Sequence[Sequence[float]],
    reference_front: Sequence[Sequence[float]],
) -> float:
    approx = [list(map(float, p)) for p in approximation]
    refs = [list(map(float, p)) for p in reference_front]
    if not refs:
        return 0.0
    if not approx:
        return float("inf")
    total = 0.0
    for r in refs:
        best = float("inf")
        for a in approx:
            d = math.sqrt(sum(max(a_j - r_j, 0.0) ** 2 for a_j, r_j in zip(a, r)))
            best = min(best, d)
        total += best
    return total / len(refs)
