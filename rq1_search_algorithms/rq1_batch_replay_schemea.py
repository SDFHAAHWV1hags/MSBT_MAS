#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SEED_RE = re.compile(r"_seed(\d+)$")

REPLAY_SCRIPTS = [
    ("NSGA-II", "rq1_nsga2_replay.py", []),
    ("SPEA2", "rq1_spea2_replay.py", []),
    ("Random", "rq1_objective_guided_random_replay.py", []),
]


def _campaign_seed(name: str) -> int:
    m = SEED_RE.search(name)
    return int(m.group(1)) if m else 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint_root", type=Path)
    p.add_argument("--runs-per-algorithm", type=int, default=20)
    p.add_argument(
        "--campaigns-per-model", type=int, default=5,
        help="how many schemeA campaigns to take per MAS (ascending by seed)",
    )
    p.add_argument("--checkpoints", default="0,5,10,20")
    p.add_argument("--models", nargs="*", default=None,
                   help="only process the given model_ids (default: all under models/selected)")
    p.add_argument("--execute", action="store_true")
    p.add_argument(
        "--output-root", type=Path,
        default=ROOT / "results" / "rq1_runs_schemea",
    )
    a = p.parse_args()

    cnums = [int(x) for x in a.checkpoints.split(",")]
    model_files = sorted((ROOT / "models" / "selected").glob("*.json"))
    if a.models:
        wanted = set(a.models)
        model_files = [m for m in model_files if m.stem in wanted]

    for model in model_files:
        mid = model.stem
        dirs = sorted(
            (
                d
                for d in a.checkpoint_root.iterdir()
                if d.is_dir() and d.name.startswith(mid + "_schemeA_")
            ),
            key=lambda d: _campaign_seed(d.name),
        )[: a.campaigns_per_model]
        if not dirs:
            print(
                f"# WARNING: {mid}: no schemeA campaign dirs under checkpoint root",
                file=sys.stderr,
            )
        for ci, cdir in enumerate(dirs, 1):
            for n in cnums:
                ck = cdir / f"C{n:03d}.json"
                if not ck.exists():
                    print(f"# WARNING: missing {ck}", file=sys.stderr)
                    continue
                for seed in range(1, a.runs_per_algorithm + 1):
                    for alg, script, extra in REPLAY_SCRIPTS:
                        out = (
                            a.output_root
                            / mid
                            / f"campaign{ci}"
                            / f"C{n:03d}"
                            / f"{alg}_seed{seed:02d}.json.gz"
                        )
                        out.parent.mkdir(parents=True, exist_ok=True)
                        cmd = [
                            sys.executable,
                            str(ROOT / script),
                            str(model),
                            str(ck),
                            "--seed", str(seed),
                            "--output", str(out),
                        ] + extra
                        print(" ".join(map(str, cmd)))
                        if a.execute:
                            subprocess.run(cmd, check=True, cwd=ROOT)


if __name__ == "__main__":
    main()
