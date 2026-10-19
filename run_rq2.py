#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parent
CAPTAIN_TREE = "rq2_test_generation_captain"
M1_TREE = "rq2_test_generation_magentic_one"

PACKS = {
    "captain_a": {
        "tree": CAPTAIN_TREE,
        "systems": {
            "CaptainAgent-1": [1, 2, 3, 4, 5],
            "CaptainAgent-10": [1, 2, 3, 4, 5],
            "CaptainAgent-49": [1, 2, 3, 4, 5],
        },
    },
    "captain_b": {
        "tree": CAPTAIN_TREE,
        "systems": {
            "CaptainAgent-101": [1, 2, 3, 4, 5],
            "CaptainAgent-106": [1, 2, 3, 4, 5],
            "CaptainAgent-117": [1, 2, 3, 4, 5],
        },
    },
    "m1": {
        "tree": M1_TREE,
        "systems": {"M1": [1, 2, 3, 4, 5]},
    },
}

METHODS = {
    "schemeA":   ("Scheme A",   "scheme_a.py",     "schemeA",   "schemeA_30.json",
                  "schemeA", "schemea"),
    "random3":   ("OG-Random",  "og_random.py",    "random3",   "random3_30.json",
                  "random3_brs04", "schemea"),
    "mb_random": ("MB-Random",  "mb_random.py",    "mb_random", "mb_random_30.json",
                  "mb_random_brs04", None),
    "c1v2":      ("LLM-Direct", "llm_direct.py",   "c1v2",      "direct_llm_v2_30.json",
                  "directllmv2", None),
}


def system_model(system: str) -> str:
    return "Magentic-One" if system == "M1" else system


def run(cmd, cwd, env=None, check=True):
    print("\n$ " + " ".join(map(str, cmd)), flush=True)
    return subprocess.run([str(x) for x in cmd], cwd=cwd, env=env, check=check)


def formal_lane(formal: str, seed: int) -> str:
    return f"{formal}_s{seed}"


def execute_lane(tree_dir, system, key, seed, num_tests, matrix):
    formal, entry, prefix, result_json, cid_token, variant = METHODS[key]
    model_id = system_model(system)
    cid = f"{model_id}_{cid_token}_deepseek_seed{seed}"
    logs_dir = matrix / "logs" / f"{prefix}_{model_id}_logs"
    result_path = matrix / result_json
    model_file = tree_dir / "models" / "selected" / f"{model_id}.json"

    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    env["MAS_RUN_LOGS_DIR"] = str(logs_dir)
    if variant:
        env["MAS_OBJECTIVE_VARIANT"] = variant

    cmd = [sys.executable, entry, model_file,
           "--llm-config", "configs/llm_configs.json",
           "--llm-id", "deepseek",
           "--executor", "executors/mas_plugin.py",
           "--num-tests", num_tests, "--seed", seed,
           "--campaign-id", cid,
           "--checkpoint-root", matrix / "checkpoints",
           "--output", result_path]
    if key != "c1v2":
        cmd += ["--gaia-root", tree_dir / "data" / "GAIA" / "2023",
                "--provision-dir", matrix / "provision"]
    if num_tests < 30 and key == "random3":
        cmd += ["--fitness-evaluations", "200"]
    run(cmd, cwd=tree_dir, env=env)
    return cid, result_path, logs_dir


def package_lane(matrix, final, cid, result_path, logs_dir):
    ck_dir = matrix / "checkpoints" / cid
    final.mkdir(parents=True, exist_ok=True)
    shutil.copy2(result_path, final / "result_30.json")
    plans = ck_dir / "plans.json"
    if plans.exists():
        shutil.copy2(plans, final / "plans.json")
    n = 0
    for tdir in sorted(logs_dir.glob("test_*")):
        mas = tdir / "mas"
        if not mas.is_dir():
            continue
        dst = final / "tests" / tdir.name.replace("test_", "t")
        dst.mkdir(parents=True, exist_ok=True)
        for f in mas.iterdir():
            shutil.copy2(f, dst / f.name)
        n += 1
    print(f"  [pack] {final.name}: result_30.json + plans.json + {n} test bundles")
    return n


def annotate_lane(tree_dir, matrix, cid, system, formal, lane_name, n_tests,
                  ann_dir):
    ann_py = PKG / "annotation" / "mast_annotate.py"
    seed = int(lane_name.rsplit("_s", 1)[1])
    ok = 0
    for t in range(1, n_tests + 1):
        tdir = next(iter(matrix.glob("logs/*_logs"))) / f"test_{t:02d}"
        mas, request = tdir / "mas", tdir / "worker_request.json"
        ck = matrix / "checkpoints" / cid / f"C{t:03d}.json"
        out = ann_dir / f"s{seed}_t{t:02d}.json"
        if not (mas.is_dir() and request.exists() and ck.exists()):
            print(f"  [ann][SKIP] {lane_name} t{t:02d}: bundle/request/checkpoint missing")
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        r = run([sys.executable, ann_py,
                 "--logs", mas, "--request", request, "--checkpoint", ck,
                 "--out", out, "--case-id", f"{lane_name}/test_{t:02d}",
                 "--llm-config", tree_dir / "configs" / "llm_configs.json"],
                cwd=PKG, check=False)
        ok += (r.returncode == 0)
    print(f"  [ann] {lane_name}: {ok}/{n_tests} tests annotated -> {ann_dir}")


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pack", choices=list(PACKS),
                    help="delivery package (default: run all)")
    ap.add_argument("--systems", nargs="*", default=None)
    ap.add_argument("--methods", nargs="*", default=list(METHODS),
                    choices=list(METHODS))
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--num-tests", type=int, default=30)
    ap.add_argument("--smoke", action="store_true",
                    help="M1 x LLM-Direct x s1 x 1 test, outputs go to results_smoke/")
    ap.add_argument("--no-annotate", action="store_true")
    ap.add_argument("--annotate-only", action="store_true",
                    help="do not re-run; only pack and annotate existing matrix artifacts")
    ap.add_argument("--out-base", default=None,
                    help="output root name (default results; use results_smoke for testing)")
    args = ap.parse_args()

    if args.smoke:
        args.pack, args.systems = "m1", ["M1"]
        args.methods, args.seeds, args.num_tests = ["c1v2"], [1], 1

    packs = [args.pack] if args.pack else list(PACKS)
    if args.out_base:
        out_base = PKG / args.out_base / "rq2"
    else:
        out_base = PKG / ("results_smoke/rq2" if args.smoke else "results/rq2")

    for pack in packs:
        cfg = PACKS[pack]
        tree_dir = PKG / cfg["tree"]
        out_root = out_base / pack
        systems = args.systems or list(cfg["systems"])
        for system in systems:
            seeds = args.seeds or cfg["systems"][system]
            for key in args.methods:
                formal = METHODS[key][0]
                for seed in seeds:
                    lane_name = f"{METHODS[key][2]}_s{seed}"
                    matrix = out_root / "matrix" / lane_name
                    cid = f"{system_model(system)}_{METHODS[key][4]}_deepseek_seed{seed}"
                    result_json = matrix / METHODS[key][3]
                    print(f"\n===== {pack} | {system} | {formal} | seed {seed} "
                          f"| {args.num_tests} tests =====")
                    if not args.annotate_only:
                        if result_json.exists():
                            print("  [skip] results already exist (delete the corresponding matrix lane to re-run)")
                        else:
                            execute_lane(tree_dir, system, key, seed,
                                         args.num_tests, matrix)
                    if result_json.exists():
                        logs_dir = matrix / "logs" / f"{METHODS[key][2]}_{system_model(system)}_logs"
                        final = out_root / "lanes" / system / formal_lane(formal, seed)
                        package_lane(matrix, final, cid, result_json, logs_dir)
                        if not args.no_annotate:
                            annotate_lane(tree_dir, matrix, cid, system, formal,
                                          lane_name, args.num_tests,
                                          out_root / "annotations" / system / formal)

    print(f"\nRQ2 done, output root: {out_base}")


if __name__ == "__main__":
    main()
