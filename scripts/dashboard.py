import sys
import argparse


def main():
    parser = argparse.ArgumentParser(description="MSA AI Dashboard & Tools")
    parser.add_argument("--dashboard", action="store_true", help="Show learning dashboard")
    parser.add_argument("--tools", action="store_true", help="Show tool usage (mafft/muscle/esm2/trimal)")
    parser.add_argument("--strategy-table", action="store_true", help="Show strategy table by problem type (n>=3)")
    parser.add_argument("--learning-curve", action="store_true", help="Show learning curve over time")
    parser.add_argument("--strategies", action="store_true", help="Show strategy performance")
    parser.add_argument("--timeline", action="store_true", help="Show learning timeline")
    parser.add_argument("--report", action="store_true", help="Generate text report")
    parser.add_argument("--export", type=str, help="Export knowledge base to file")
    parser.add_argument("--import", dest="import_file", type=str, help="Import knowledge base from file")
    parser.add_argument("--clean", type=int, nargs="?", const=10, help="Clean old experiments (keep N)")
    parser.add_argument("--validate", action="store_true", help="Cross-validate strategies")
    parser.add_argument("--feedback", type=str, help="Add feedback for experiment (experiment_id)")
    parser.add_argument("--format-info", action="store_true", help="Show storage format info (TOON/sparse/JSON)")
    parser.add_argument("--migrate-to-toon", action="store_true", help="Migrate JSONL knowledge base to TOON + sparse")
    parser.add_argument("--sparse-info", action="store_true", help="Show sparse index details")
    parser.add_argument("--toon-context", action="store_true", help="Preview TOON context sent to LLM")
    parser.add_argument("--compare-formats", action="store_true", help="Compare storage formats (size, records)")
    parser.add_argument("--inspect", type=str, nargs="?", const="", help="Inspect experiment details by ID (omit ID to list all)")

    args = parser.parse_args()

    if args.dashboard:
        from visualization import (
            print_dashboard, print_tool_usage, print_strategy_table,
            print_learning_curve, print_esm_gap_stats, print_esm_post_stats,
        )
        print_dashboard()
        print_tool_usage()
        print_strategy_table()
        print_learning_curve()
        print_esm_gap_stats()
        print_esm_post_stats()
    elif args.tools:
        from visualization import print_tool_usage, print_esm_gap_stats, print_esm_post_stats
        print_tool_usage()
        print_esm_gap_stats()
        print_esm_post_stats()
    elif args.strategy_table:
        from visualization import print_strategy_table
        print_strategy_table()
    elif args.learning_curve:
        from visualization import print_learning_curve
        print_learning_curve()
    elif args.strategies:
        from visualization import print_strategy_performance
        print_strategy_performance()
    elif args.timeline:
        from visualization import print_learning_timeline
        print_learning_timeline()
    elif args.report:
        from visualization import export_text_report
        export_text_report()
    elif args.export:
        from export_import import export_knowledge
        export_knowledge(args.export)
    elif args.import_file:
        from export_import import import_knowledge
        import_knowledge(args.import_file, merge=True)
    elif args.clean is not None:
        from export_import import clean_knowledge
        result = clean_knowledge(args.clean)
        print(result)
    elif args.validate:
        from rag_advanced import cross_validate, load_experiments
        experiments = load_experiments()
        result = cross_validate(experiments)
        if result.get("message"):
            print(result["message"])
        else:
            print(f"Folds: {result.get('folds')} | test n={result.get('test_size')}")
            print(f"Strategy-aware accuracy: {result.get('accuracy', 0)*100:.1f}%")
            print(f"Baseline accuracy:       {result.get('baseline_accuracy', 0)*100:.1f}%")
            print(f"Lift over baseline:      {result.get('lift', 0)*100:+.1f} pts")
            print(f"Brier score (lower=better): {result.get('brier_score', 'n/a')}")
            print(f"Gap-improvement MAE:         {result.get('gap_mae', 'n/a')}")
            print("\nPer problem type:")
            from visualization import _gap_set_stats, _fmt_gap_block, _sorted_problems
            by_problem = {}
            for e in experiments:
                by_problem.setdefault(e.get("problem_type", "unknown"), []).append(e)
            for p in _sorted_problems(by_problem):
                d = result.get("per_problem", {}).get(p, {})
                acc = d.get("accuracy", 0) * 100 if d else 0
                n = d.get("n", len(by_problem[p])) if d else len(by_problem[p])
                print(f"  ▸ {p}  n={n}  acc={acc:.1f}%")
                for line in _fmt_gap_block(_gap_set_stats(by_problem[p]), indent="    "):
                    print(line)
            print("\nSample predictions (2 per problem type):")
            samples = result.get("results", [])
            order = ["high_gap_fraction", "moderate_gap_fraction", "good_quality"]
            problems = [p for p in order if any(r["problem_type"] == p for r in samples)]
            problems += sorted(
                {r["problem_type"] for r in samples} - set(order)
            )
            for p in problems:
                print(f"  ▸ {p}")
                shown = 0
                for r in samples:
                    if r["problem_type"] != p:
                        continue
                    print(
                        f"    {r['experiment_id']}: base={r['baseline_rate']} "
                        f"strat={r['strategy_rate']} -> {r['actual_outcome']} "
                        f"| strategy={r['recommended_strategy']}"
                    )
                    shown += 1
                    if shown >= 2:
                        break
    elif args.feedback:
        from feedback_system import interactive_feedback
        interactive_feedback(args.feedback)
    elif args.format_info:
        from format_tools import cmd_format_info
        cmd_format_info()
    elif args.migrate_to_toon:
        from format_tools import cmd_migrate_to_toon
        cmd_migrate_to_toon()
    elif args.sparse_info:
        from format_tools import cmd_sparse_info
        cmd_sparse_info()
    elif args.toon_context:
        from format_tools import cmd_toon_context
        cmd_toon_context()
    elif args.compare_formats:
        from format_tools import cmd_compare_formats
        cmd_compare_formats()
    elif args.inspect is not None:
        from format_tools import cmd_inspect
        cmd_inspect(args.inspect if args.inspect else None)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()