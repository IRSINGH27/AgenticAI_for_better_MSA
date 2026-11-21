#!/usr/bin/env python3
"""
Comparison test: Default MAFFT vs Default MUSCLE vs Our AI Pipeline
Generates random protein sequences and runs all three methods.
"""

import sys
import os
import time
import random
import pathlib

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))

from msa_runner import run_msa
from muscle_runner import run_muscle
from extended_metrics import compute_extended_metrics
from pipeline_loop import run_pipeline

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
NUM_SEQUENCES = 50
SEQ_LENGTH = 200
SEED = 42


def generate_random_protein(num_seqs: int, seq_length: int, seed: int) -> str:
    """Generate random protein sequences in FASTA format."""
    random.seed(seed)
    fasta_lines = []
    for i in range(num_seqs):
        seq = "".join(random.choices(AMINO_ACIDS, k=seq_length))
        fasta_lines.append(f">random_seq_{i+1}")
        for j in range(0, seq_length, 80):
            fasta_lines.append(seq[j:j+80])
    return "\n".join(fasta_lines) + "\n"


def run_mafft_default(fasta_text: str) -> dict:
    """Run default MAFFT and return alignment + metrics + time."""
    print("  Running MAFFT --auto ...", flush=True)
    start = time.time()
    alignment = run_msa(fasta_text, {"strategy": "--auto"})
    elapsed = time.time() - start
    metrics = compute_extended_metrics(alignment)
    return {"name": "MAFFT (default)", "alignment": alignment, "metrics": metrics, "time": elapsed, "iters": 1}


def run_muscle_default(fasta_text: str) -> dict:
    """Run default MUSCLE and return alignment + metrics + time."""
    print("  Running MUSCLE align ...", flush=True)
    start = time.time()
    alignment = run_muscle(fasta_text, {"algorithm": "align"})
    elapsed = time.time() - start
    metrics = compute_extended_metrics(alignment)
    return {"name": "MUSCLE (default)", "alignment": alignment, "metrics": metrics, "time": elapsed, "iters": 1}


def run_our_pipeline(fasta_text: str) -> dict:
    """Run our AI pipeline and return alignment + metrics + time."""
    print("  Running Our AI Pipeline ...", flush=True)
    start = time.time()
    result = run_pipeline(
        fasta_text=fasta_text,
        max_iterations=10,
        convergence_tolerance=0.001,
        convergence_patience=3,
        warm_start_threshold=0.8
    )
    elapsed = time.time() - start
    return {
        "name": "Our AI Pipeline",
        "alignment": result["final_alignment"],
        "metrics": result["final_metrics"],
        "time": elapsed,
        "iters": result["iterations"],
        "exp_id": result["experiment_id"]
    }


def format_metrics_table(results: list) -> str:
    """Format results as a comparison table."""
    lines = []
    header = f"{'Method':<25} {'Gap%':>6} {'SP':>6} {'Identity':>8} {'Hill_q1':>8} {'Iters':>6} {'Time':>8}"
    lines.append("=" * len(header))
    lines.append("COMPARISON RESULTS")
    lines.append("=" * len(header))
    lines.append(header)
    lines.append("-" * len(header))

    for r in results:
        m = r["metrics"]
        gap = f"{m['gap_fraction']:.4f}"
        sp = f"{m['sp_score']:.4f}"
        identity = f"{m['identity']:.4f}"
        hill = f"{m['hill_q1_avg']:.2f}"
        iters = f"{r['iters']}"
        t = f"{r['time']:.1f}s"
        lines.append(f"{r['name']:<25} {gap:>6} {sp:>6} {identity:>8} {hill:>8} {iters:>6} {t:>8}")

    lines.append("-" * len(header))

    # Find best for each metric
    best_gap = min(results, key=lambda x: x["metrics"]["gap_fraction"])
    best_sp = max(results, key=lambda x: x["metrics"]["sp_score"])
    best_id = max(results, key=lambda x: x["metrics"]["identity"])
    best_hill = min(results, key=lambda x: x["metrics"]["hill_q1_avg"])
    best_time = min(results, key=lambda x: x["time"])

    lines.append(f"\nBest gap_fraction:  {best_gap['name']} ({best_gap['metrics']['gap_fraction']:.4f})")
    lines.append(f"Best sp_score:      {best_sp['name']} ({best_sp['metrics']['sp_score']:.4f})")
    lines.append(f"Best identity:      {best_id['name']} ({best_id['metrics']['identity']:.4f})")
    lines.append(f"Best hill_q1_avg:   {best_hill['name']} ({best_hill['metrics']['hill_q1_avg']:.2f})")
    lines.append(f"Fastest:            {best_time['name']} ({best_time['time']:.1f}s)")

    lines.append("\n" + "=" * len(header))
    lines.append("DETAILED METRICS")
    lines.append("=" * len(header))

    for r in results:
        lines.append(f"\n--- {r['name']} ---")
        for k, v in r["metrics"].items():
            if k != "hill_problematic_columns":
                if isinstance(v, float):
                    lines.append(f"  {k}: {v:.4f}")
                else:
                    lines.append(f"  {k}: {v}")

    return "\n".join(lines)


def main():
    print("=" * 60)
    print("COMPARISON TEST: MAFFT vs MUSCLE vs Our Pipeline")
    print(f"Sequences: {NUM_SEQUENCES} random proteins, {SEQ_LENGTH} residues each")
    print("=" * 60)

    # Generate sequences
    print("\n[1/4] Generating random protein sequences...", flush=True)
    fasta_text = generate_random_protein(NUM_SEQUENCES, SEQ_LENGTH, SEED)
    fasta_path = SCRIPT_DIR.parent / "test_comparison.fasta"
    with open(fasta_path, "w") as f:
        f.write(fasta_text)
    print(f"  Saved to {fasta_path}")

    results = []

    # Run MAFFT
    print("\n[2/4] Running default MAFFT...", flush=True)
    try:
        r = run_mafft_default(fasta_text)
        results.append(r)
        print(f"  Done: gap={r['metrics']['gap_fraction']:.4f}, time={r['time']:.1f}s")
    except Exception as e:
        print(f"  FAILED: {e}")

    # Run MUSCLE
    print("\n[3/4] Running default MUSCLE...", flush=True)
    try:
        r = run_muscle_default(fasta_text)
        results.append(r)
        print(f"  Done: gap={r['metrics']['gap_fraction']:.4f}, time={r['time']:.1f}s")
    except Exception as e:
        print(f"  FAILED: {e}")

    # Run our pipeline
    print("\n[4/4] Running Our AI Pipeline...", flush=True)
    try:
        r = run_our_pipeline(fasta_text)
        results.append(r)
        print(f"  Done: gap={r['metrics']['gap_fraction']:.4f}, iters={r['iters']}, time={r['time']:.1f}s, exp={r['exp_id']}")
    except Exception as e:
        print(f"  FAILED: {e}")

    # Format and display results
    if len(results) >= 2:
        table = format_metrics_table(results)
        print("\n" + table)

        # Save results
        results_path = SCRIPT_DIR.parent / "test_comparison_results.txt"
        with open(results_path, "w") as f:
            f.write(f"Comparison Test: {NUM_SEQUENCES} random proteins, {SEQ_LENGTH} residues\n")
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(table)
        print(f"\nResults saved to {results_path}")
    else:
        print("\nNot enough successful runs to compare.")


if __name__ == "__main__":
    main()
