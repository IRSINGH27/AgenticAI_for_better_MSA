#!/usr/bin/env python3
"""
Comparison test: MAFFT vs MUSCLE vs Our AI Pipeline
Using real Pfam families (50 sequences each, sampled from families)
Pipeline uses MAFFT + MUSCLE (ESM-2 skipped for speed)
"""
import sys
import time
import random
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent.resolve()))

from msa_runner import run_msa
from muscle_runner import run_muscle
from extended_metrics import compute_extended_metrics
from pipeline_loop import run_pipeline

FAMILIES_DIR = pathlib.Path(__file__).parent.parent / "data" / "pfam" / "families"
TEST_FAMILIES = ["PF00007", "PF00002", "PF00010", "PF00001", "PF00005"]
NUM_SEQS = 50
SEED = 42
MAX_ITERATIONS = 100
CONVERGENCE_TOLERANCE = 0.001
CONVERGENCE_PATIENCE = 3


def load_fasta(filepath):
    sequences = {}
    current = None
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                current = line[1:]
                sequences[current] = ""
            elif current is not None:
                sequences[current] += line
    return sequences


def subsample_fasta(filepath, n=50, seed=42):
    sequences = load_fasta(filepath)
    rng = random.Random(seed)
    keys = list(sequences.keys())
    if len(keys) > n:
        keys = rng.sample(keys, n)
    lines = []
    for k in keys:
        lines.append(f">{k}")
        lines.append(sequences[k])
    return "\n".join(lines)


def count_seqs(filepath):
    count = 0
    with open(filepath) as f:
        for line in f:
            if line.startswith(">"):
                count += 1
    return count


def fmt_metrics(m):
    return (f"gap={m['gap_fraction']:.4f}  sp={m['sp_score']:.4f}  "
            f"identity={m['identity']:.4f}  hill_q1={m['hill_q1_avg']:.2f}")


def main():
    results_file = pathlib.Path(__file__).parent.parent / "test_comparison_pfam_results.txt"
    out = open(results_file, "w")

    def log(msg=""):
        print(msg, flush=True)
        out.write(msg + "\n")

    log("=" * 70)
    log("COMPARISON TEST: MAFFT vs MUSCLE vs Our AI Pipeline")
    log(f"Families: {TEST_FAMILIES}")
    log(f"Sequences per family: {NUM_SEQS} (seed={SEED})")
    log(f"Pipeline: max_iterations={MAX_ITERATIONS}, patience={CONVERGENCE_PATIENCE}")
    log("=" * 70)

    all_results = []

    for family in TEST_FAMILIES:
        faa_path = FAMILIES_DIR / f"{family}.faa"
        if not faa_path.exists():
            log(f"\n!!! {family}.faa not found, skipping")
            continue

        total_seqs = count_seqs(str(faa_path))
        fasta_text = subsample_fasta(str(faa_path), n=NUM_SEQS, seed=SEED)

        log(f"\n{'=' * 70}")
        log(f"FAMILY: {family} ({total_seqs} total seqs, {NUM_SEQS} sampled)")
        log(f"{'=' * 70}")

        # MAFFT
        log(f"\n--- MAFFT --auto ---")
        t0 = time.time()
        aln_mafft = run_msa(fasta_text, {"strategy": "--auto"})
        t_mafft = time.time() - t0
        m_mafft = compute_extended_metrics(aln_mafft)
        log(f"  {fmt_metrics(m_mafft)}  time={t_mafft:.1f}s")

        # MUSCLE
        log(f"\n--- MUSCLE align ---")
        t0 = time.time()
        aln_muscle = run_muscle(fasta_text, {"algorithm": "align"})
        t_muscle = time.time() - t0
        m_muscle = compute_extended_metrics(aln_muscle)
        log(f"  {fmt_metrics(m_muscle)}  time={t_muscle:.1f}s")

        # Pipeline
        log(f"\n--- Our AI Pipeline ({MAX_ITERATIONS} iterations) ---")
        t0 = time.time()
        result = run_pipeline(
            fasta_text=fasta_text,
            max_iterations=MAX_ITERATIONS,
            convergence_tolerance=CONVERGENCE_TOLERANCE,
            convergence_patience=CONVERGENCE_PATIENCE,
            warm_start_threshold=0.8,
        )
        t_pipe = time.time() - t0
        m_pipe = result["final_metrics"]
        log(f"  FINAL: {fmt_metrics(m_pipe)}  time={t_pipe:.1f}s")
        log(f"  Iterations: {result['iterations']}")
        log(f"  Experiment ID: {result['experiment_id']}")

        from collections import Counter
        actions = result.get("actions_tried", [])
        action_counts = Counter(a.get("action", a) if isinstance(a, dict) else a for a in actions)
        log(f"  Actions: {dict(action_counts)}")

        log(f"\n  SUMMARY ({family}):")
        log(f"  {'Method':<25} {'Gap%':>6} {'SP':>6} {'Identity':>8} {'Hill_q1':>8} {'Time':>8}")
        log(f"  {'-'*60}")
        log(f"  {'MAFFT --auto':<25} {m_mafft['gap_fraction']:>6.4f} {m_mafft['sp_score']:>6.4f} {m_mafft['identity']:>8.4f} {m_mafft['hill_q1_avg']:>8.2f} {t_mafft:>7.1f}s")
        log(f"  {'MUSCLE align':<25} {m_muscle['gap_fraction']:>6.4f} {m_muscle['sp_score']:>6.4f} {m_muscle['identity']:>8.4f} {m_muscle['hill_q1_avg']:>8.2f} {t_muscle:>7.1f}s")
        log(f"  {'Our Pipeline':<25} {m_pipe['gap_fraction']:>6.4f} {m_pipe['sp_score']:>6.4f} {m_pipe['identity']:>8.4f} {m_pipe['hill_q1_avg']:>8.2f} {t_pipe:>7.1f}s")

        best_gap = min(m_mafft["gap_fraction"], m_muscle["gap_fraction"], m_pipe["gap_fraction"])
        if m_pipe["gap_fraction"] < best_gap + 0.0001:
            log(f"  ** Pipeline WINS or ties **")
        elif m_mafft["gap_fraction"] == best_gap:
            log(f"  ** MAFFT wins **")
        else:
            log(f"  ** MUSCLE wins **")

        all_results.append({
            "family": family,
            "total_seqs": total_seqs,
            "mafft": {"metrics": m_mafft, "time": t_mafft},
            "muscle": {"metrics": m_muscle, "time": t_muscle},
            "pipeline": {"metrics": m_pipe, "time": t_pipe, "iterations": result["iterations"],
                         "actions": actions},
        })

    # Overall summary
    log(f"\n{'=' * 70}")
    log("OVERALL SUMMARY")
    log(f"{'=' * 70}")
    log(f"{'Family':<10} {'MAFFT gap':>10} {'MUSCLE gap':>10} {'Pipeline gap':>12} {'Winner':>10}")
    log(f"{'-'*55}")
    wins = {"MAFFT": 0, "MUSCLE": 0, "Pipeline": 0}
    for r in all_results:
        mg = r["mafft"]["metrics"]["gap_fraction"]
        sg = r["muscle"]["metrics"]["gap_fraction"]
        pg = r["pipeline"]["metrics"]["gap_fraction"]
        best = "MAFFT" if mg <= sg and mg <= pg else ("MUSCLE" if sg <= pg else "Pipeline")
        wins[best] += 1
        log(f"{r['family']:<10} {mg:>10.4f} {sg:>10.4f} {pg:>12.4f} {best:>10}")

    log(f"\n  Wins: MAFFT={wins['MAFFT']}, MUSCLE={wins['MUSCLE']}, Pipeline={wins['Pipeline']}")
    log(f"\nResults saved to: {results_file}")
    out.close()


if __name__ == "__main__":
    main()
