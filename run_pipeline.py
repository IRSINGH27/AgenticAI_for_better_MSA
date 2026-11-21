import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "scripts"))

import argparse
from pipeline_loop import run_pipeline


def main():
    parser = argparse.ArgumentParser(description="Run MSA pipeline with AI control")
    parser.add_argument("fasta_file", nargs="?", help="Path to FASTA file")
    parser.add_argument("--max-iterations", "-n", type=int, default=10, help="Max iterations")
    parser.add_argument("--tolerance", "-t", type=float, default=0.01, help="Convergence tolerance")
    parser.add_argument("--patience", "-p", type=int, default=2, help="Convergence patience")
    parser.add_argument("--warm-start-threshold", "-w", type=float, default=0.8, help="Similarity threshold for warm start (0-1)")
    parser.add_argument("--esm-analysis", action="store_true", help="Run ESM probability analysis after alignment")
    parser.add_argument("--esm-refine-gaps", action="store_true", help="Use ESM conservation to refine gap placement")
    parser.add_argument("--esm-alpha", type=float, default=2.0, help="Gap penalty scaling factor for ESM-informed gaps")
    parser.add_argument(
        "--esm-post",
        choices=["always", "if_bad", "off"],
        default="if_bad",
        help="Post-align ESM refine: if_bad (default; only if gap still bad) | always | off",
    )
    args = parser.parse_args()

    if args.fasta_file:
        with open(args.fasta_file, "r") as f:
            fasta_text = f.read()
        print(f"Loaded: {args.fasta_file}")
    else:
        print("No FASTA file provided, using default example")
        fasta_text = None

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

    print("\n" + "=" * 40)
    print("FINAL RESULT:")
    print(f"  Iterations: {result['iterations']}")
    print(f"  Metrics: {result['final_metrics']}")
    print(f"  Experiment ID: {result['experiment_id']}")
    if result.get("esm_analysis"):
        print(f"  ESM Analysis: r={result['esm_analysis'].get('conservation_correlation', 'N/A')}")
    if result.get("esm_post_applied"):
        print(f"  ESM post: gap {result.get('esm_gap_before')} -> {result.get('esm_gap_after')}")
    print("\nAlignment preview:")
    print(result['final_alignment'][:500])


if __name__ == "__main__":
    main()