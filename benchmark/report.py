#!/usr/bin/env python3
"""Benchmark report: leaderboards (SP/TC + our metrics) and correlations.

Reads results/results.jsonl written by bench_runner.py.

Usage:
  python report.py                    # stdout tables
  python report.py --csv              # also write results/leaderboard.csv
  python report.py --corr             # correlation of our metrics vs SP
"""
import argparse
import json
import pathlib
import sys
from collections import defaultdict

HERE = pathlib.Path(__file__).resolve().parent
RESULTS_LOG = HERE / "results" / "results.jsonl"


def load_rows():
    if not RESULTS_LOG.exists():
        sys.exit(f"No results at {RESULTS_LOG} — run bench_runner.py first.")
    rows = []
    for line in RESULTS_LOG.read_text().splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return [r for r in rows if not r.get("error")]


def fmt(x, nd=3):
    return f"{x:.{nd}f}" if isinstance(x, (int, float)) else "-"


def leaderboard(rows) -> list:
    """(dataset, method) → aggregate means."""
    groups = defaultdict(list)
    for r in rows:
        groups[(r.get("dataset"), r.get("method"))].append(r)
    out = []
    for (ds, method), rs in sorted(groups.items()):
        def mean(path):
            vals = []
            for r in rs:
                v = r
                for p in path:
                    v = (v or {}).get(p) if isinstance(v, dict) else None
                    if v is None:
                        break
                if isinstance(v, (int, float)):
                    vals.append(v)
            return sum(vals) / len(vals) if vals else None
        out.append({
            "dataset": ds, "method": method, "n": len(rs),
            "SP": mean(["sp"]), "TC": mean(["tc"]),
            "gap_fraction": mean(["extended_metrics", "gap_fraction"]),
            "hill_q1_avg": mean(["extended_metrics", "hill_q1_avg"]),
            "conservation": mean(["extended_metrics", "conservation"]),
            "identity": mean(["extended_metrics", "identity"]),
            "runtime_s": mean(["runtime_s"]),
        })
    return out


def print_leaderboard(lb, as_csv: bool = False):
    cols = ["dataset", "method", "n", "SP", "TC", "gap_fraction",
            "hill_q1_avg", "conservation", "identity", "runtime_s"]
    if as_csv:
        print(",".join(cols))
        for row in lb:
            print(",".join(
                str(round(row[c], 4)) if isinstance(row[c], float)
                else str(row[c]) for c in cols))
        return
    print("=" * 96)
    print(f"{'dataset':10s} {'method':9s} {'n':>4s} {'SP':>7s} {'TC':>7s} "
          f"{'gap':>7s} {'hill_q1':>8s} {'cons':>7s} {'ident':>7s} {'sec':>7s}")
    print("-" * 96)
    for row in lb:
        print(f"{str(row['dataset']):10s} {row['method']:9s} {row['n']:>4d} "
              f"{fmt(row['SP']):>7s} {fmt(row['TC']):>7s} "
              f"{fmt(row['gap_fraction']):>7s} {fmt(row['hill_q1_avg']):>8s} "
              f"{fmt(row['conservation']):>7s} {fmt(row['identity']):>7s} "
              f"{fmt(row['runtime_s'], 1):>7s}")
    print("=" * 96)


def correlations(rows):
    """Do OUR metrics track reference SP? (validates scoring theory)"""
    try:
        import numpy as np
        from scipy.stats import pearsonr
    except ImportError:
        print("numpy/scipy required for --corr")
        return
    pairs = [("gap_fraction", []), ("hill_q1_avg", []), ("conservation", []),
             ("identity", []), ("sp_score", [])]
    sps = []
    for r in rows:
        em = r.get("extended_metrics") or {}
        if "sp" not in em and "sp" not in r:
            continue
        sps.append(r["sp"])
        for key, vals in pairs:
            vals.append(em.get(key))
    print("\nCorrelation of our metrics vs reference SP "
          f"(n={len(sps)} runs):")
    print(f"  {'metric':16s} {'pearson_r':>10s} {'p-value':>10s}")
    for key, vals in pairs:
        v = [(a, b) for a, b in zip(vals, sps) if a is not None]
        if len(v) < 3:
            continue
        x, y = zip(*v)
        r, p = pearsonr(x, y)
        print(f"  {key:16s} {r:>10.3f} {p:>10.4f}")


def show_best():
    best = HERE / "results" / "best_params.json"
    if best.exists():
        print("\n=== OPTUNA BEST PARAMS ===")
        print(best.read_text())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", action="store_true")
    ap.add_argument("--corr", action="store_true")
    args = ap.parse_args()
    rows = load_rows()
    lb = leaderboard(rows)
    print_leaderboard(lb, as_csv=args.csv)
    if args.corr:
        correlations(rows)
    show_best()
    if args.csv:
        out = HERE / "results" / "leaderboard.csv"
        cols = ["dataset", "method", "n", "SP", "TC", "gap_fraction",
                "hill_q1_avg", "conservation", "identity", "runtime_s"]
        with out.open("w") as f:
            f.write(",".join(cols) + "\n")
            for row in lb:
                f.write(",".join(
                    str(round(row[c], 4)) if isinstance(row[c], float)
                    else str(row[c]) for c in cols) + "\n")
        print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
