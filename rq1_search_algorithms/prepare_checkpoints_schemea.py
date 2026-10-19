#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MAS_OBJECTIVE_VARIANT", "schemea")

ROOT = Path(__file__).resolve().parent
PACKAGE_ROOT = ROOT.parent
DATA_ROOT = Path(os.environ.get("MAS_DATA_ROOT", PACKAGE_ROOT / "data"))
sys.path[:0] = [str(ROOT / "core")]

from adaptive_search import (
    CampaignArchive,
    ExecutedTest,
    Gene,
    ModelRuntime,
    coverage_features,
)
from checkpoint_io import serialize_archive, sha256_json

LANE_GLOBS = ["inputs/schemea_lanes/*/s*/plans.json"]

SYSTEM_TO_MODEL = {"M1": "Magentic-One"}

CUTOFFS = (0, 5, 10, 20)
EXPECTED_PLANS = 30
SEED_RE = re.compile(r"_?s(?:eed)?(\d+)$")


def genes_from_dicts(rows):
    out = []
    for x in rows or []:
        params = tuple(
            sorted((str(k), str(v)) for k, v in (x.get("params") or {}).items())
        )
        out.append(Gene(str(x.get("rule_id", "")), str(x.get("event", "")), params))
    return out


def build_planning_archive(plans, k, runtime):
    arc = CampaignArchive()
    for p in plans[:k]:
        genes = genes_from_dicts(p["selected_path"])
        arc.covered_features |= coverage_features(runtime, genes)
        arc.executed_tests.append(
            ExecutedTest(list(genes), "", list(genes), "PLANNED_TARGET", [])
        )
    return arc


def _seed_of(cid: str) -> int:
    m = SEED_RE.search(cid)
    return int(m.group(1)) if m else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="only list discovered campaigns and exit")
    ap.add_argument("--out-root", type=Path,
                    default=ROOT / "results" / "rq1_checkpoint_root_schemea")
    ap.add_argument("--data-root", type=Path, default=DATA_ROOT)
    ap.add_argument("--checkpoints", default=",".join(map(str, CUTOFFS)))
    args = ap.parse_args()

    cutoffs = tuple(int(x) for x in args.checkpoints.split(","))

    found = {}
    for pattern in LANE_GLOBS:
        for p in sorted(args.data_root.glob(pattern)):
            lane = p.parent
            system = lane.parent.name
            m = SEED_RE.search(lane.name)
            if not m:
                continue
            mid = SYSTEM_TO_MODEL.get(system, system)
            seed = int(m.group(1))
            cid = f"{mid}_schemeA_deepseek_seed{seed}"
            found.setdefault(cid, p.resolve())
    campaigns = dict(sorted(found.items(), key=lambda kv: (kv[0].split("_schemeA_")[0],
                                                           _seed_of(kv[0]))))

    by_model = {}
    for cid in campaigns:
        mid = cid.split("_schemeA_", 1)[0]
        by_model.setdefault(mid, []).append(cid)

    print(f"data root: {args.data_root}")
    print(f"found {len(campaigns)} Scheme-A campaigns across {len(by_model)} MASes:")
    for mid in sorted(by_model):
        print(f"  {mid}: {len(by_model[mid])}  campaigns seeds={sorted(_seed_of(c) for c in by_model[mid])}")

    if args.list:
        for cid, p in campaigns.items():
            print(f"    {cid} -> {p}")
        return

    model_cache = {}

    def get_model(mid):
        if mid not in model_cache:
            mp = ROOT / "models" / "selected" / f"{mid}.json"
            if not mp.exists():
                raise SystemExit(f"[FATAL] frozen model file not found: {mp}")
            model = json.loads(mp.read_text(encoding="utf-8"))
            model_cache[mid] = (mp, model, ModelRuntime(model))
        return model_cache[mid]

    made = 0
    for cid, plans_path in campaigns.items():
        mid = cid.split("_schemeA_", 1)[0]
        model_path, model, runtime = get_model(mid)
        model_hash = sha256_json(model)

        saved = json.loads(plans_path.read_text(encoding="utf-8"))
        plans = saved.get("plans", [])
        if len(plans) != EXPECTED_PLANS:
            raise SystemExit(
                f"[FATAL] {cid}: plans.json contains {len(plans)} plans, "
                f"expected {EXPECTED_PLANS}"
            )

        orig_ck = plans_path.parent / "C030.json"
        if not orig_ck.exists():
            orig_ck = plans_path.parent / "C000.json"
        if orig_ck.exists():
            orig = json.loads(orig_ck.read_text(encoding="utf-8"))
            if orig.get("model_hash") not in (None, model_hash):
                raise SystemExit(
                    f"[FATAL] {cid}: model_hash of original checkpoint {orig_ck.name} "
                    f"does not match {model_path.name}; refusing to generate"
                )

        seed = _seed_of(cid)
        out_dir = args.out_root / cid
        out_dir.mkdir(parents=True, exist_ok=True)

        for k in cutoffs:
            arc = build_planning_archive(plans, k, runtime)
            ck = {
                "checkpoint_schema": "1.0",
                "campaign_id": cid,
                "model_id": model.get("model_id"),
                "model_hash": model_hash,
                "metadata": {
                    "method": "SchemeA-TargetPlanning-BRSv04",
                    "source": "synthetic_planning_archive_from_plans_json",
                    "plans_json": str(plans_path),
                    "active_objectives": [
                        "coverage_gain", "behavioral_diversity", "path_cost",
                    ],
                    "archive_semantics": "selected_target_paths_only",
                    "runtime_feedback_used_for_search": False,
                    "seed": seed,
                },
                "test_index": k,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "archive": serialize_archive(arc),
                "search_parameters": {},
                "last_round": {
                    "kind": "planning_archive",
                    "planned_targets": k,
                },
            }
            (out_dir / f"C{k:03d}.json").write_text(
                json.dumps(ck, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            made += 1
        print(f"  [ok] {cid}: {len(cutoffs)}  planning-archive checkpoints")

    print(f"\ndone: {len(campaigns)} campaigns x {len(cutoffs)} cutoffs = {made} checkpoints")
    print(f"output root: {args.out_root}")
    print("dry-run check:")
    print(
        "  MAS_OBJECTIVE_VARIANT=schemea /home/zs/mas/.venv/bin/python "
        "rq1_batch_replay_schemea.py results/rq1_checkpoint_root_schemea | head"
    )


if __name__ == "__main__":
    main()
