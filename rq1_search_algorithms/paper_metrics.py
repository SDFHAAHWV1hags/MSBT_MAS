#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / "core")]

from rq1_common import (
    empirical_bounds,
    hypervolume_min,
    igd_plus,
    load_json_any,
    nd_min,
    normalize_points,
)
from adaptive_search import OBJECTIVE_DIRECTIONS
try:
    from adaptive_search import ACTIVE_OBJECTIVE_INDICES
except ImportError:
    ACTIVE_OBJECTIVE_INDICES = tuple(range(len(OBJECTIVE_DIRECTIONS)))

PASS_OUTCOMES = {"PASS", "SUCCESS", "OK", "MOCK_SUCCESS"}


def _write_rows(rows: List[Dict[str, Any]], output: Path, extra_json: Optional[Dict[str, Any]] = None):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output.with_suffix(".csv")
    json_path = output.with_suffix(".json")
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        fields = list(rows[0].keys()) if rows else []
        writer = csv.DictWriter(f, fieldnames=fields)
        if fields:
            writer.writeheader()
            writer.writerows(rows)
    payload = {"rows": rows}
    if extra_json:
        payload.update(extra_json)
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(csv_path)
    print(json_path)


def _failure_signature(row: Dict[str, Any]) -> Optional[str]:
    sig = row.get("failure_signature")
    if sig is not None and str(sig).strip():
        return str(sig).strip()
    outcome = str(row.get("outcome", "")).strip()
    if outcome and outcome.upper() not in PASS_OUTCOMES:
        return "OUTCOME:" + outcome
    oracle = row.get("oracle_verdict")
    if isinstance(oracle, dict) and str(oracle.get("verdict", "")).upper() == "FAIL":
        return "ORACLE:FAIL"
    return None


def _load_failure_type_map(path: Optional[Path]) -> Dict[str, str]:
    if path is None:
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "mapping" in data:
        data = data["mapping"]
    if not isinstance(data, dict):
        raise ValueError("Failure type map must be a JSON object or {'mapping': {...}}.")
    return {str(k): str(v) for k, v in data.items()}


def _failure_type(row: Dict[str, Any], sig: Optional[str], mapping: Dict[str, str]) -> Optional[str]:
    value = row.get("failure_type")
    if value is not None and str(value).strip():
        return str(value).strip()
    if sig and sig in mapping:
        return mapping[sig]
    return None


def _execution_time_from_timings(timings: Dict[str, Any]) -> float:
    for key in ("execution_sec", "mas_execution_sec", "executor_sec"):
        if timings.get(key) is not None:
            return float(timings[key] or 0.0)
    return float(timings.get("total_sec", 0.0) or 0.0)


def _test_total_time(row: Dict[str, Any]) -> float:
    if row.get("test_total_sec") is not None:
        return float(row.get("test_total_sec") or 0.0)

    timings = row.get("timings", {}) or {}
    generation = float(row.get("nlp_generation_sec", 0.0) or 0.0)
    if generation == 0.0:
        generation = float(timings.get("generation_sec", 0.0) or 0.0)
    search = float(row.get("search_sec", 0.0) or 0.0)
    execution = _execution_time_from_timings(timings)
    return search + generation + execution


def campaign_metrics(campaign: Path, failure_type_map: Optional[Path] = None) -> Dict[str, Any]:
    data = json.loads(Path(campaign).read_text(encoding="utf-8"))
    tests = list(data.get("tests", []))
    mapping = _load_failure_type_map(failure_type_map)

    signatures: List[str] = []
    failure_types: List[str] = []
    missing_type_count = 0
    total_time_sec = 0.0

    for row in tests:
        total_time_sec += _test_total_time(row)
        sig = _failure_signature(row)
        if not sig:
            continue
        signatures.append(sig)
        ftype = _failure_type(row, sig, mapping)
        if ftype is None:
            missing_type_count += 1
        else:
            failure_types.append(ftype)

    uf = len(set(signatures))
    ft_available = (missing_type_count == 0) or not signatures
    ft = len(set(failure_types)) if ft_available else None
    hours = total_time_sec / 3600.0

    return {
        "campaign_file": str(campaign),
        "model_id": data.get("model_id"),
        "method": data.get("method"),
        "llm_id": data.get("llm_id"),
        "seed": data.get("seed"),
        "tests": len(tests),
        "unique_failures": uf,
        "distinct_failure_types": ft,
        "failure_type_labels_complete": ft_available,
        "failures_missing_type_label": missing_type_count,
        "total_campaign_time_sec": total_time_sec,
        "unique_failures_per_hour": (uf / hours) if hours > 0 else None,
        "failure_types_per_hour": (ft / hours) if hours > 0 and ft is not None else None,
    }


def campaign_command(args):
    row = campaign_metrics(args.campaign, args.failure_type_map)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
    print(args.output)


def campaign_batch_command(args):
    files = sorted(args.campaign_dir.rglob("*.json"))
    rows = []
    for f in files:
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(d, dict) or "tests" not in d or "method" not in d:
            continue
        rows.append(campaign_metrics(f, args.failure_type_map))
    _write_rows(rows, args.output)


def _bounds_nd(sets: Sequence[Sequence[Sequence[float]]], ndim: int) -> Dict[str, List[float]]:
    points = [list(map(float, p)) for fs in sets for p in fs]
    if not points:
        raise ValueError("Cannot compute empirical bounds from zero points.")
    mins = [min(p[j] for p in points) for j in range(ndim)]
    maxs = [max(p[j] for p in points) for j in range(ndim)]
    return {"mins": mins, "maxs": maxs}


def _normalize_nd(
    points: Sequence[Sequence[float]],
    bounds: Dict[str, Sequence[float]],
    directions: Sequence[int],
) -> List[List[float]]:
    mins = list(map(float, bounds["mins"]))
    maxs = list(map(float, bounds["maxs"]))
    out: List[List[float]] = []
    for p in points:
        q: List[float] = []
        for j, (value, direction) in enumerate(zip(map(float, p), directions)):
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


def rq1_context_key(d: Dict[str, Any]) -> Tuple[str, str, int]:
    campaign_id = d.get("checkpoint_campaign_id")
    if not campaign_id:
        checkpoint = Path(str(d.get("checkpoint", "")))
        campaign_id = checkpoint.parent.name
    return (
        str(d.get("model_id")),
        str(campaign_id),
        int(d.get("checkpoint_test_index", -1)),
    )


def _cap_front(points: Sequence[Sequence[float]], cap: int) -> List[List[float]]:
    pts = [list(map(float, p)) for p in points]
    if cap is None or cap <= 0 or len(pts) <= cap:
        return pts
    d = len(pts[0])
    lo = [min(p[j] for p in pts) for j in range(d)]
    hi = [max(p[j] for p in pts) for j in range(d)]
    rng = [(hi[j] - lo[j]) or 1.0 for j in range(d)]

    def norm(p):
        return [(p[j] - lo[j]) / rng[j] for j in range(d)]

    npts = [norm(p) for p in pts]
    chosen: set = set()
    for j in range(d):
        k = min(range(len(pts)), key=lambda i, j=j: (npts[i][j], i))
        chosen.add(k)
    sel = sorted(chosen)

    def ed(i, s):
        return math.sqrt(sum((npts[i][a] - npts[s][a]) ** 2 for a in range(d)))

    dist = [min(ed(i, s) for s in sel) for i in range(len(pts))]
    while len(sel) < cap:
        nxt = -1
        best = -1.0
        for i in range(len(pts)):
            if i in chosen:
                continue
            if dist[i] > best:
                best, nxt = dist[i], i
        if nxt < 0:
            break
        chosen.add(nxt)
        sel.append(nxt)
        for i in range(len(pts)):
            dd = ed(i, nxt)
            if dd < dist[i]:
                dist[i] = dd
    return [pts[i] for i in sorted(sel)]


def rq1_command(args):
    files = sorted(list(args.runs_dir.rglob("*.json")) + list(args.runs_dir.rglob("*.json.gz")))
    groups: Dict[Tuple[str, str, int], List[Tuple[Path, Dict[str, Any]]]] = defaultdict(list)

    for f in files:
        try:
            d = load_json_any(f)
        except Exception:
            continue
        if d.get("schema") not in {"rq2-search-run-v1", "rq2-search-run-v2"}:
            continue
        groups[rq1_context_key(d)].append((f, d))

    rows: List[Dict[str, Any]] = []
    contexts: Dict[str, Any] = {}

    for key, runs in sorted(groups.items()):
        if getattr(args, "active_only", False):
            indices = list(ACTIVE_OBJECTIVE_INDICES)
            objective_space = f"active{len(indices)}"
        else:
            indices = list(range(len(OBJECTIVE_DIRECTIONS)))
            objective_space = f"all{len(indices)}"
        directions = [OBJECTIVE_DIRECTIONS[j] for j in indices]

        raw_sizes: List[int] = []
        sets = []
        for _, d in runs:
            raw = [x["objectives"] for x in d.get("final_pareto_front", [])]
            raw_sizes.append(len(raw))
            capped = _cap_front(raw, getattr(args, "cap_front", None))
            sets.append([[float(p[j]) for j in indices] for p in capped])
        if not any(sets):
            continue

        bounds = _bounds_nd(sets, len(indices))
        normalized_sets = [_normalize_nd(s, bounds, directions) for s in sets]
        reference_front = nd_min([p for s in normalized_sets for p in s])
        hv_reference = [args.reference_point] * len(indices)
        context_id = f"{key[0]}::{key[1]}::C{key[2]:03d}"
        contexts[context_id] = {
            "model_id": key[0],
            "checkpoint_campaign_id": key[1],
            "checkpoint_test_index": key[2],
            "objective_space": objective_space,
            "objective_indices": indices,
            "bounds": bounds,
            "empirical_reference_front": reference_front,
            "hv_reference_point": hv_reference,
        }

        for (f, d), points, raw_n in zip(runs, normalized_sets, raw_sizes):
            rows.append(
                {
                    "context_id": context_id,
                    "model_id": d.get("model_id"),
                    "checkpoint_campaign_id": key[1],
                    "checkpoint_test_index": key[2],
                    "algorithm": d.get("algorithm"),
                    "seed": d.get("seed"),
                    "fitness_evaluations": d.get("fitness_evaluations"),
                    "runtime_sec": d.get("runtime_sec"),
                    "front_size": len(points),
                    "raw_front_size": raw_n,
                    "objective_space": objective_space,
                    "hypervolume": hypervolume_min(points, hv_reference),
                    "igd_plus": igd_plus(points, reference_front),
                    "run_file": str(f),
                }
            )

    _write_rows(rows, args.output, {"contexts": contexts})


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)

    c = sub.add_parser("campaign")
    c.add_argument("campaign", type=Path)
    c.add_argument("--failure-type-map", type=Path, default=None)
    c.add_argument("--output", type=Path, required=True)
    c.set_defaults(func=campaign_command)

    cb = sub.add_parser("campaign-batch")
    cb.add_argument("campaign_dir", type=Path)
    cb.add_argument("--failure-type-map", type=Path, default=None)
    cb.add_argument("--output", type=Path, required=True)
    cb.set_defaults(func=campaign_batch_command)

    r = sub.add_parser("rq1")
    r.add_argument("runs_dir", type=Path)
    r.add_argument("--reference-point", type=float, default=1.1)
    r.add_argument("--cap-front", type=int, default=0,
                   help="cap every run front to this many points (0 = raw fronts)")
    r.add_argument("--active-only", action="store_true",
                   help="compute HV/IGD+ only on the ACTIVE_OBJECTIVE_INDICES active objectives "
                        "(Scheme-A three-objective setting, drop constant compatibility slots)")
    r.add_argument("--output", type=Path, required=True)
    r.set_defaults(func=rq1_command)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
