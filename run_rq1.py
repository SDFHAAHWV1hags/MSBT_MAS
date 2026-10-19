#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parent
TREE = PKG / "rq1_search_algorithms"
PY = sys.executable

MODEL_TAGS = {
    "Magentic-One": "mag",
    "CaptainAgent-1": "cap1",
    "CaptainAgent-10": "cap10",
    "CaptainAgent-49": "cap4",
    "CaptainAgent-101": "cap101",
    "CaptainAgent-106": "cap106",
    "CaptainAgent-117": "cap117",
}


def run(cmd: list[str], cwd: Path, env: dict | None = None):
    print("\n$ " + " ".join(map(str, cmd)), flush=True)
    subprocess.run([str(x) for x in cmd], cwd=cwd, env=env, check=True)


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true",
                    help="run only the first Magentic-One campaign with 1 replay seed")
    ap.add_argument("--models", nargs="*", default=None,
                    help="only process the given models (file names under models/selected, without .json)")
    ap.add_argument("--campaigns-per-model", type=int, default=5)
    ap.add_argument("--runs-per-algorithm", type=int, default=20)
    ap.add_argument("--checkpoints", default="0,5,10,20")
    ap.add_argument("--step", choices=["prepare", "replay", "metrics"],
                    help="run only one step (default: run all steps in order)")
    ap.add_argument("--no-metrics", action="store_true")
    args = ap.parse_args()

    out_root = PKG / ("results_smoke/rq1" if args.smoke else "results/rq1")
    ck_root = out_root / "checkpoint_root"
    runs_root = out_root / "rq1_runs_schemea"
    metrics_root = out_root / "metrics_schemea"
    models = ["Magentic-One"] if args.smoke else (args.models or list(MODEL_TAGS))
    n_campaigns = 1 if args.smoke else args.campaigns_per_model
    n_runs = 1 if args.smoke else args.runs_per_algorithm

    env = dict(os.environ, MAS_OBJECTIVE_VARIANT="schemea",
               PYTHONDONTWRITEBYTECODE="1")

    if args.step in (None, "prepare"):
        run([PY, "prepare_checkpoints_schemea.py",
             "--data-root", PKG / "data",
             "--out-root", ck_root], cwd=TREE)

    if args.step in (None, "replay"):
        cmd = [PY, "rq1_batch_replay_schemea.py", ck_root,
               "--execute",
               "--campaigns-per-model", n_campaigns,
               "--runs-per-algorithm", n_runs,
               "--checkpoints", args.checkpoints,
               "--models", *models,
               "--output-root", runs_root]
        run(cmd, cwd=TREE, env=env)

    if args.step in (None, "metrics") and not args.no_metrics:
        metrics_root.mkdir(parents=True, exist_ok=True)
        for mid in models:
            stem = f"rq1_metrics_schemea_{MODEL_TAGS.get(mid, mid)}"
            run([PY, "rq1_metrics_per_context.py",
                 runs_root, metrics_root / stem,
                 "--models", mid, "--active-only"], cwd=TREE, env=env)

    print(f"\nRQ1 done, output root: {out_root}")


if __name__ == "__main__":
    main()
