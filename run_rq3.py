#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parent
TREE = PKG / "rq3_objective_ablation"
MODEL = TREE / "models" / "selected" / "Magentic-One.json"

VARIANTS = {
    "sa_covdiv": ("covdiv", "covdiv"),
    "sa_covcost": ("covcost", "covcost"),
    "sa_divcost": ("divcost", "divcost"),
}


def run(cmd, cwd, env=None, check=True):
    print("\n$ " + " ".join(map(str, cmd)), flush=True)
    return subprocess.run([str(x) for x in cmd], cwd=cwd, env=env, check=check)


def execute(variant, seed, num_tests, matrix):
    short, cid_token = VARIANTS[variant]
    lane = f"{short}_s{seed}"
    lane_dir = matrix / lane
    cid = f"Magentic-One_schemeA_{cid_token}_deepseek_seed{seed}"
    result_path = lane_dir / "schemeA_30.json"
    logs_root = lane_dir / "mblogs"

    env = dict(os.environ,
               MAS_OBJECTIVE_VARIANT=variant,
               MAS_RUN_LOGS_DIR=str(logs_root),
               PYTHONDONTWRITEBYTECODE="1")
    cmd = [sys.executable, "scheme_a.py", MODEL,
           "--llm-config", "configs/llm_configs.json",
           "--llm-id", "deepseek",
           "--executor", "executors/mas_plugin.py",
           "--num-tests", num_tests, "--seed", seed,
           "--campaign-id", cid,
           "--checkpoint-root", lane_dir / "checkpoints",
           "--output", result_path,
           "--gaia-root", TREE / "data" / "GAIA" / "2023",
           "--provision-dir", lane_dir / "provision"]
    if not result_path.exists():
        run(cmd, cwd=TREE, env=env)
    else:
        print(f"  [skip] {lane}: schemeA_30.json already exists")
    return lane, cid, result_path, logs_root


def package(lane, cid, result_path, logs_root, lanes_root):
    ck_dir = result_path.parent / "checkpoints" / cid
    final = lanes_root / lane
    final.mkdir(parents=True, exist_ok=True)
    shutil.copy2(result_path, final / "result_30.json")
    plans = ck_dir / "plans.json"
    if plans.exists():
        shutil.copy2(plans, final / "plans.json")
    n = 0
    for tdir in sorted(logs_root.glob("test_*")):
        mas = tdir / "mas"
        if not mas.is_dir():
            continue
        dst = final / "tests" / tdir.name.replace("test_", "t")
        dst.mkdir(parents=True, exist_ok=True)
        for f in mas.iterdir():
            shutil.copy2(f, dst / f.name)
        n += 1
    print(f"  [pack] {lane}: result_30.json + plans.json + {n} test bundles")


def annotate(lane, cid, seed, num_tests, lane_matrix_dir, ann_root):
    ann_py = PKG / "annotation" / "mast_annotate.py"
    short = lane.split("_s")[0]
    ann_dir = ann_root / short
    ok = 0
    for t in range(1, num_tests + 1):
        tdir = lane_matrix_dir / "mblogs" / f"test_{t:02d}"
        mas, request = tdir / "mas", tdir / "worker_request.json"
        ck = lane_matrix_dir / "checkpoints" / cid / f"C{t:03d}.json"
        out = ann_dir / f"s{seed}_t{t:02d}.json"
        if not (mas.is_dir() and request.exists() and ck.exists()):
            print(f"  [ann][SKIP] {lane} t{t:02d}: missing artifacts")
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        r = run([sys.executable, ann_py,
                 "--logs", mas, "--request", request, "--checkpoint", ck,
                 "--out", out, "--case-id", f"{lane}/test_{t:02d}",
                 "--llm-config", TREE / "configs" / "llm_configs.json"],
                cwd=PKG, check=False)
        ok += (r.returncode == 0)
    print(f"  [ann] {lane}: {ok}/{num_tests} tests annotated -> {ann_dir}")


def metrics(lane, result_path, metrics_root):
    metrics_root.mkdir(parents=True, exist_ok=True)
    out = metrics_root / f"{lane}.json"
    run([sys.executable, "campaign_metrics.py", result_path, MODEL,
         "--output", out], cwd=TREE)


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS),
                    choices=list(VARIANTS))
    ap.add_argument("--seeds", nargs="*", type=int, default=[1, 2, 3, 4, 5])
    ap.add_argument("--num-tests", type=int, default=30)
    ap.add_argument("--smoke", action="store_true",
                    help="covdiv x s1 x 1 test, outputs go to results_smoke/")
    ap.add_argument("--no-annotate", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.variants, args.seeds, args.num_tests = ["sa_covdiv"], [1], 1

    out_root = PKG / ("results_smoke/rq3" if args.smoke else "results/rq3")
    matrix_root, lanes_root = out_root / "matrix", out_root / "lanes"
    ann_root, metrics_root = out_root / "annotations", out_root / "metrics"

    for variant in args.variants:
        for seed in args.seeds:
            short = VARIANTS[variant][0]
            print(f"\n===== RQ3 {variant} | seed {seed} | {args.num_tests} tests =====")
            lane, cid, result_path, logs_root = execute(
                variant, seed, args.num_tests, matrix_root)
            package(lane, cid, result_path, logs_root, lanes_root)
            if not args.no_annotate:
                annotate(lane, cid, seed, args.num_tests,
                         matrix_root / lane, ann_root)
            metrics(lane, result_path, metrics_root)

    print(f"\nRQ3 done, output root: {out_root}")


if __name__ == "__main__":
    main()
