#!/usr/bin/env python3
"""Run the MSA pipeline on one FASTA; write full result JSON to --out.

Runs under the AI conda env (torch/transformers). stdout is ignored (pipeline
prints progress); the machine-readable result goes to the --out file so the
bench harness (separate env) can read it.
"""
import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from pipeline_loop import run_pipeline  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("-n", "--max-iterations", type=int, default=10)
    ap.add_argument("-t", "--tolerance", type=float, default=0.01)
    ap.add_argument("-p", "--patience", type=int, default=2)
    ap.add_argument("-w", "--warm-start-threshold", type=float, default=0.8)
    ap.add_argument("--esm-analysis", action="store_true")
    ap.add_argument("--esm-refine-gaps", action="store_true")
    ap.add_argument("--esm-alpha", type=float, default=2.0)
    ap.add_argument("--esm-post", choices=["always", "if_bad", "off"], default="if_bad")
    args = ap.parse_args()

    fasta_text = pathlib.Path(args.fasta).read_text()
    result = run_pipeline(
        fasta_text=fasta_text,
        max_iterations=args.max_iterations,
        convergence_tolerance=args.tolerance,
        convergence_patience=args.patience,
        warm_start_threshold=args.warm_start_threshold,
        esm_analysis=args.esm_analysis,
        esm_refine_gaps=args.esm_refine_gaps,
        esm_alpha=args.esm_alpha,
        esm_post=args.esm_post,
    )
    payload = {
        "alignment": result.get("final_alignment", ""),
        "final_metrics": result.get("final_metrics", {}),
        "iterations": result.get("iterations"),
        "experiment_id": result.get("experiment_id"),
        "was_trimmed": result.get("was_trimmed"),
        "esm_post_applied": result.get("esm_post_applied"),
        "esm_gap_before": result.get("esm_gap_before"),
        "esm_gap_after": result.get("esm_gap_after"),
    }
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload))
    return 0


if __name__ == "__main__":
    sys.exit(main())
