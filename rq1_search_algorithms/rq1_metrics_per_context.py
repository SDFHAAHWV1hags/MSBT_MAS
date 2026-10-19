#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

SUITE = Path(__file__).resolve().parent
PY = sys.executable


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs_root")
    ap.add_argument("out_stem")
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--cap-front", type=int, default=0)
    ap.add_argument("--active-only", action="store_true",
                    help="compute metrics only on active objective dimensions (Scheme-A three-objective setting); "
                         "automatically sets MAS_OBJECTIVE_VARIANT=schemea for subprocesses")
    args = ap.parse_args()

    root = Path(args.runs_root).resolve()
    out_csv = Path(args.out_stem + ".csv")
    out_json = Path(args.out_stem + ".json")
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    ctx_dirs = sorted({p.parent for p in root.rglob("*.json.gz")})
    if args.models:
        ctx_dirs = [d for d in ctx_dirs if any(m in d.parts for m in args.models)]

    all_rows: list[dict] = []
    contexts: dict = {}
    stage = root.parent / "_ctx_stage"

    for i, ctx in enumerate(ctx_dirs, 1):
        if stage.exists():
            shutil.rmtree(stage)
        stage.mkdir(parents=True)
        for gz in ctx.glob("*.json.gz"):
            os.link(gz, stage / gz.name)

        tmp_out = stage / "out"
        env = dict(os.environ)
        cmd = [PY, "paper_metrics.py", "rq1", str(stage), "--output", str(tmp_out)]
        if args.cap_front and args.cap_front > 0:
            cmd += ["--cap-front", str(args.cap_front)]
        if args.active_only:
            cmd += ["--active-only"]
            env.setdefault("MAS_OBJECTIVE_VARIANT", "schemea")
        r = subprocess.run(
            cmd,
            cwd=SUITE, env=env, capture_output=True, text=True,
        )
        if r.returncode != 0:
            print(f"[FAIL] {ctx}: {r.stderr[-500:]}")
            continue

        with open(tmp_out.with_suffix(".csv"), newline="", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        with open(tmp_out.with_suffix(".json"), encoding="utf-8") as f:
            j = json.load(f)
        all_rows.extend(rows)
        contexts.update(j.get("contexts", {}))
        cid = rows[0]["context_id"] if rows else str(ctx)
        print(f"[{i}/{len(ctx_dirs)}] {cid}: {len(rows)} runs", flush=True)

    shutil.rmtree(stage, ignore_errors=True)

    fields = ["context_id","model_id","checkpoint_campaign_id","checkpoint_test_index",
              "algorithm","seed","fitness_evaluations","runtime_sec","front_size",
              "raw_front_size","objective_space","hypervolume","igd_plus","run_file"]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in all_rows:
            w.writerow({k: r.get(k, "") for k in fields})
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"rows": all_rows, "contexts": contexts}, f)
    print(f"\nmerged: {len(all_rows)} rows, {len(contexts)} contexts -> {out_csv}")


if __name__ == "__main__":
    main()
