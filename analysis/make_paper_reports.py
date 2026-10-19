#!/usr/bin/env python3
import argparse
import csv
import gzip
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MAS_OBJECTIVE_VARIANT", "schemea")

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "rq1_search_algorithms"),
                str(ROOT / "rq1_search_algorithms" / "core")]

from rq1_common import hypervolume_min, igd_plus, load_json_any, nd_min
from paper_metrics import _bounds_nd, _cap_front, _normalize_nd
from adaptive_search import ACTIVE_OBJECTIVE_INDICES, OBJECTIVE_DIRECTIONS
from scipy import stats

METHODS = ["Scheme A", "OG-Random", "MB-Random", "LLM-Direct"]
RQ2_PACKS = {
    "m1": ["M1"],
    "captain_a": ["CaptainAgent-1", "CaptainAgent-10", "CaptainAgent-49"],
    "captain_b": ["CaptainAgent-101", "CaptainAgent-106", "CaptainAgent-117"],
}
RQ3_VARIANTS = ["covdiv", "covcost", "divcost"]
RQ3_BASELINE = "schemea"
CUTOFFS = [0, 5, 10, 20]
ALGS = ["NSGA-II", "SPEA2", "Random"]
FM_KEYS = [f"FM-1.{i}" for i in range(1, 6)] + \
          [f"FM-2.{i}" for i in range(1, 7)] + \
          [f"FM-3.{i}" for i in range(1, 4)]


def a12(x, y):
    n, m = len(x), len(y)
    if n == 0 or m == 0:
        return float("nan")
    vals = sorted([(v, 0) for v in x] + [(v, 1) for v in y])
    ranks = {}
    i = 0
    rank_of = [0.0] * (n + m)
    while i < len(vals):
        j = i
        while j < len(vals) and vals[j][0] == vals[i][0]:
            j += 1
        avg = (i + 1 + j) / 2.0
        for k in range(i, j):
            rank_of[k] = avg
        i = j
    rx = sum(rank_of[k] for k, (_, g) in enumerate(vals) if g == 0)
    return (rx - n * (n + 1) / 2.0) / (n * m)


def rq1_context_metrics(ctx_dir):
    runs = []
    for alg in ALGS:
        for f in sorted(ctx_dir.glob(f"{alg}_seed*.json.gz")):
            d = load_json_any(f)
            raw = [p["objectives"] for p in d.get("final_pareto_front", [])]
            runs.append([alg, raw])
    if not runs:
        return {}
    indices = list(ACTIVE_OBJECTIVE_INDICES)
    directions = [OBJECTIVE_DIRECTIONS[j] for j in indices]
    sets = [[[float(p[j]) for j in indices] for p in _cap_front(raw, None)]
            for _, raw in runs]
    bounds = _bounds_nd(sets, len(indices))
    norm = [_normalize_nd(s, bounds, directions) for s in sets]
    ref = nd_min([p for s in norm for p in s])
    hv_ref = [1.1] * len(indices)
    out = defaultdict(lambda: {"hv": [], "igd": []})
    for (alg, _), points in zip(runs, norm):
        out[alg]["hv"].append(hypervolume_min(points, hv_ref))
        out[alg]["igd"].append(igd_plus(points, ref))
    return out


def build_rq1(results_root):
    runs_root = results_root / "rq1" / "rq1_runs_schemea"
    data = defaultdict(lambda: defaultdict(lambda: {"hv": [], "igd": []}))
    for model_dir in sorted(p for p in runs_root.iterdir() if p.is_dir()):
        for camp in range(1, 6):
            for cp in CUTOFFS:
                ctx = model_dir / f"campaign{camp}" / f"C{cp:03d}"
                if not ctx.is_dir():
                    print(f"  MISSING {ctx}")
                    continue
                m = rq1_context_metrics(ctx)
                for alg in ALGS:
                    for metric in ("hv", "igd"):
                        data[(model_dir.name, cp)][alg][metric].extend(
                            m.get(alg, {}).get(metric, []))
    rows = []
    for (model, cp), algs in sorted(data.items()):
        counts = {a: len(algs[a]["hv"]) for a in ALGS}
        row = {"system": model, "CP": cp, "n_per_alg": counts[ALGS[0]]}
        for x, y, tag in (("NSGA-II", "Random", "N/R"),
                          ("SPEA2", "Random", "S/R"),
                          ("NSGA-II", "SPEA2", "N/S")):
            row[f"{tag}_HV_A12"] = a12(algs[x]["hv"], algs[y]["hv"])
            row[f"{tag}_IGD_A12"] = 1.0 - a12(algs[x]["igd"], algs[y]["igd"])
        rows.append(row)
    return rows


def iter_rq2_cases(results_root):
    for pack, systems in RQ2_PACKS.items():
        base = results_root / "rq2" / pack
        for system in systems:
            for method in METHODS:
                for seed in range(1, 6):
                    for test in range(1, 31):
                        t = f"t{test:02d}"
                        ann_f = (base / "annotations" / system / method
                                 / f"s{seed}_{t}.json")
                        lane_f = (base / "lanes" / system
                                  / f"{method}_s{seed}" / "tests" / t
                                  / "llm_calls.json")
                        if not ann_f.is_file():
                            print(f"  MISSING ann {ann_f}")
                            continue
                        ann = json.loads(ann_f.read_text(encoding="utf-8"))
                        fms = ann.get("fms", {})
                        fms = {k: v for k, v in fms.items() if k in FM_KEYS}
                        hit_set = {k for k, v in fms.items()
                                   if isinstance(v, dict) and v.get("hit")}
                        usage = None
                        if lane_f.is_file():
                            u = json.loads(lane_f.read_text(encoding="utf-8"))
                            if (u.get("total_llm_wall_ms") or 0) > 0:
                                usage = u
                        yield (system, method, seed, test,
                               bool(hit_set), hit_set, usage)


def _inputs_schemea_usage(results_root, seed, test):
    f = (ROOT / "data" / "inputs" / "schemea_lanes" / "M1"
         / f"s{seed}" / "result_30.json")
    if not f.is_file():
        return None
    d = json.loads(f.read_text(encoding="utf-8"))
    entry = next((t for t in d.get("tests", [])
                  if int(t.get("test_index", -1)) == test), None)
    if not entry:
        return None
    u = entry.get("mas_llm_usage")
    if u and (u.get("total_llm_wall_ms") or 0) > 0:
        return u
    return None


def iter_rq3_cases(results_root):
    base = results_root / "rq3"
    for variant in [RQ3_BASELINE] + RQ3_VARIANTS:
        for seed in range(1, 6):
            for test in range(1, 31):
                t = f"t{test:02d}"
                ann_f = base / "annotations" / variant / f"s{seed}_{t}.json"
                if not ann_f.is_file():
                    print(f"  MISSING ann {ann_f}")
                    continue
                ann = json.loads(ann_f.read_text(encoding="utf-8"))
                fms = {k: v for k, v in ann.get("fms", {}).items()
                       if k in FM_KEYS}
                hit_set = {k for k, v in fms.items()
                           if isinstance(v, dict) and v.get("hit")}
                if variant == RQ3_BASELINE:
                    usage = _inputs_schemea_usage(results_root, seed, test)
                else:
                    usage = None
                    llm_f = (base / "lanes" / f"{variant}_s{seed}"
                             / "tests" / t / "llm_calls.json")
                    if llm_f.is_file():
                        u = json.loads(llm_f.read_text(encoding="utf-8"))
                        if (u.get("total_llm_wall_ms") or 0) > 0:
                            usage = u
                yield variant, seed, test, bool(hit_set), hit_set, usage


def summarize_cases(cases):
    n = len(cases)
    n_err = sum(c[0] for c in cases)
    fm_union = set().union(*(c[1] for c in cases)) if cases else set()
    usages = [c[2] for c in cases if c[2] is not None]
    tot_tokens = sum(u.get("total_tokens", 0) for u in usages)
    tot_wall_s = sum(u.get("total_llm_wall_ms", 0) for u in usages) / 1000.0
    tot_calls = sum(u.get("num_calls", 0) for u in usages)
    return {
        "n_cases": n,
        "n_usage_cases": len(usages),
        "N_FM_rate": n_err / n if n else float("nan"),
        "n_fm_cases": n_err,
        "C_FM": len(fm_union),
        "tokens_per_test_k": (tot_tokens / len(usages) / 1000.0
                              if usages else float("nan")),
        "E_FM_per_1M": n_err / tot_tokens * 1e6 if tot_tokens else float("nan"),
        "calls_per_test": tot_calls / len(usages) if usages else float("nan"),
        "llm_s_per_test": tot_wall_s / len(usages) if usages else float("nan"),
        "fm_per_1k_s": n_err / tot_wall_s * 1000.0 if tot_wall_s else float("nan"),
        "_hits": [int(c[0]) for c in cases],
    }


def build_rq2(results_root):
    buckets = defaultdict(list)
    per_seed = defaultdict(lambda: defaultdict(list))
    for system, method, seed, test, hit, hit_set, usage in iter_rq2_cases(results_root):
        buckets[(system, method)].append((hit, hit_set, usage))
        per_seed[(system, method)][seed].append(int(hit))
    order = [s for _, ss in RQ2_PACKS.items() for s in ss]
    baselines = ["OG-Random", "MB-Random", "LLM-Direct"]
    metrics, stats_rows, all_rates = {}, [], {}
    for system in order:
        groups = {m: summarize_cases(buckets[(system, m)]) for m in METHODS}
        metrics[system] = groups
        rates = {m: [sum(per_seed[(system, m)][s]) / len(per_seed[(system, m)][s])
                     for s in sorted(per_seed[(system, m)])]
                 for m in METHODS}
        kw_H, kw_p = stats.kruskal(*[rates[m] for m in METHODS])
        row = {"system": system, "kw_p": kw_p}
        for b in baselines:
            a, y = rates["Scheme A"], rates[b]
            row[f"{b}_A12"] = a12(a, y)
            w = sum(x > z for x, z in zip(a, y))
            t = sum(x == z for x, z in zip(a, y))
            l = sum(x < z for x, z in zip(a, y))
            row[f"{b}_WTL"] = f"{w}-{t}-{l}"
            row[f"{b}_W"], row[f"{b}_T"], row[f"{b}_L"] = w, t, l
        stats_rows.append(row)
        all_rates[system] = rates
    return metrics, stats_rows, all_rates


def build_rq3(results_root):
    buckets = defaultdict(list)
    for variant, seed, test, hit, hit_set, usage in iter_rq3_cases(results_root):
        buckets[variant].append((hit, hit_set, usage))
    return {v: summarize_cases(buckets[v])
            for v in [RQ3_BASELINE] + RQ3_VARIANTS}


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})




def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT / "results"))
    args = ap.parse_args()
    results_root = Path(args.root)
    out_dir = results_root / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("RQ1: recomputing HV/IGD+ per context from 8400 gz ...")
    rq1 = build_rq1(results_root)
    write_csv(out_dir / "rq1_a12.csv", rq1,
              ["system", "CP", "n_per_alg",
               "N/R_HV_A12", "N/R_IGD_A12",
               "S/R_HV_A12", "S/R_IGD_A12",
               "N/S_HV_A12", "N/S_IGD_A12"])

    print("RQ2: collecting 7 systems x 4 methods x 150 cases ...")
    rq2_metrics, rq2_stats, rq2_rates = build_rq2(results_root)
    flat = []
    for system, groups in rq2_metrics.items():
        for m, g in groups.items():
            flat.append({"system": system, "method": m,
                         **{k: v for k, v in g.items() if not k.startswith("_")}})
    write_csv(out_dir / "rq2_metrics.csv", flat,
              ["system", "method", "n_cases", "n_usage_cases",
               "n_fm_cases", "N_FM_rate", "C_FM",
               "tokens_per_test_k", "E_FM_per_1M"])
    t5_fields = ["system", "kw_p"]
    for b in ("OG-Random", "MB-Random", "LLM-Direct"):
        t5_fields += [f"{b}_A12", f"{b}_W", f"{b}_T", f"{b}_L", f"{b}_WTL"]
    write_csv(out_dir / "rq2_table5_campaign.csv", rq2_stats, t5_fields)
    rate_rows = [{"system": s, "method": m,
                  **{f"seed{i+1}_rate": v[i] for i in range(5)}}
                 for s, mm in rq2_rates.items() for m, v in mm.items()]
    write_csv(out_dir / "rq2_campaign_rates.csv", rate_rows,
              ["system", "method"] + [f"seed{i}_rate" for i in range(1, 6)])

    print("RQ3: Scheme A baseline (literal seeds 1-5) + 3 variants, "
          "4 x 150 cases ...")
    rq3 = build_rq3(results_root)
    write_csv(out_dir / "rq3_metrics.csv",
              [{"variant": v, **{k: x for k, x in g.items()
                                 if not k.startswith("_")}}
               for v, g in rq3.items()],
              ["variant", "n_cases", "n_usage_cases",
               "n_fm_cases", "N_FM_rate", "C_FM", "calls_per_test",
               "llm_s_per_test", "tokens_per_test_k", "fm_per_1k_s"])

    print(f"done -> {out_dir}")


if __name__ == "__main__":
    main()
