import json
import math
import os
import statistics
from collections import Counter, defaultdict
from typing import Dict, List, Optional
import pathlib

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
KNOWLEDGE_DIR = SCRIPT_DIR.parent / "knowledge"
LOGS_DIR = SCRIPT_DIR.parent / "logs"

PROBLEM_ORDER = ["high_gap_fraction", "moderate_gap_fraction", "good_quality"]


def _problem_key(problem: str) -> int:
    try:
        return PROBLEM_ORDER.index(problem)
    except ValueError:
        return len(PROBLEM_ORDER)


def _sorted_problems(problems) -> List[str]:
    return sorted(problems, key=_problem_key)


def _is_reference(problem: str) -> bool:
    return problem == "good_quality"


def _mean(values: List[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _median(values: List[float]) -> float:
    return statistics.median(values) if values else 0.0


def _gap_set_stats(experiments: List[Dict]) -> Dict:
    """Mean/median of init gap, final gap, and % gap reduction over a set.

    reduction% = (init - final) / init * 100, skipped when init == 0.
    """
    inits, finals, reductions = [], [], []
    for e in experiments:
        init = float(e.get("initial_gap_fraction", 0) or 0)
        final = float(e.get("final_gap_fraction", 0) or 0)
        inits.append(init)
        finals.append(final)
        if init > 0:
            reductions.append((init - final) / init * 100)
    return {
        "n": len(experiments),
        "init_mean": _mean(inits),
        "init_med": _median(inits),
        "final_mean": _mean(finals),
        "final_med": _median(finals),
        "red_mean": _mean(reductions),
        "red_med": _median(reductions),
        "red_n": len(reductions),
    }


def _fmt_gap_block(s: Dict, indent: str = "      ") -> List[str]:
    lines = [
        f"{indent}gap init:  mean {s['init_mean']:.3f}  med {s['init_med']:.3f}",
        f"{indent}gap final: mean {s['final_mean']:.3f}  med {s['final_med']:.3f}",
    ]
    if s["red_n"]:
        lines.append(
            f"{indent}reduction: mean {s['red_mean']:.0f}%  med {s['red_med']:.0f}%  (n={s['red_n']})"
        )
    else:
        lines.append(f"{indent}reduction: — (no initial gaps)")
    return lines


def load_experiments() -> List[Dict]:
    from rag_retrieve import load_knowledge
    return load_knowledge()


def load_feedback() -> List[Dict]:
    path = os.path.join(KNOWLEDGE_DIR, "user_feedback.json")
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        return json.load(f)


def print_dashboard() -> None:
    experiments = load_experiments()
    feedback = load_feedback()

    print("\n" + "=" * 50)
    print("       MSA AI LEARNING DASHBOARD")
    print("=" * 50)

    print(f"\n📊 TOTAL EXPERIMENTS: {len(experiments)}")

    if experiments:
        overall = _gap_set_stats(experiments)
        print(
            f"\n📉 OVERALL GAP: init mean {overall['init_mean']:.3f} → "
            f"final mean {overall['final_mean']:.3f}  "
            f"(reduction mean {overall['red_mean']:.0f}%, med {overall['red_med']:.0f}%)"
        )

        by_problem = defaultdict(list)
        for exp in experiments:
            by_problem[exp.get("problem_type", "unknown")].append(exp)

        print("\n📌 BY PROBLEM TYPE")
        for problem in _sorted_problems(by_problem):
            group = by_problem[problem]
            n = len(group)
            outcomes = Counter(e.get("final_outcome", "unknown") for e in group)
            trimmed = sum(1 for e in group if e.get("was_trimmed"))
            tag = "  [reference — any strategy works]" if _is_reference(problem) else ""
            stats = _gap_set_stats(group)
            print(f"\n  ▸ {problem} (n={n}){tag}")
            print(f"      outcomes:    {', '.join(f'{k} {v} ({v*100//n}%)' for k, v in sorted(outcomes.items()))}")
            for line in _fmt_gap_block(stats):
                print(line)
            print(f"      trimmed:     {trimmed}/{n}")

    if feedback:
        ratings = [f.get("rating", 0) for f in feedback]
        avg_rating = sum(ratings) / len(ratings) if ratings else 0
        print(f"\n⭐ USER FEEDBACK: {avg_rating:.2f}/5 ({len(feedback)} responses)")
        print("   Rating distribution:")
        for i in range(1, 6):
            count = ratings.count(i)
            bar = "█" * count
            print(f"     {i} stars: {bar} ({count})")

    print("\n" + "=" * 50)


def print_tool_usage() -> None:
    experiments = load_experiments()
    exp_problem = {e.get("experiment_id"): e.get("problem_type", "unknown") for e in experiments}

    # per-problem tool iteration counts (log file name == experiment_id)
    tool_iters = defaultdict(Counter)
    tool_strategies = defaultdict(lambda: defaultdict(Counter))

    if LOGS_DIR.is_dir():
        for path in LOGS_DIR.glob("*.jsonl"):
            problem = exp_problem.get(path.stem, "unknown")
            try:
                with open(path) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            entry = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        params = entry.get("params") or {}
                        tool = params.get("tool")
                        if not tool:
                            continue
                        tool_iters[problem][tool] += 1
                        strategy = str(params.get("strategy") or "").strip()
                        extra = str(params.get("extra_args") or "").strip()
                        combo = (strategy + " " + extra).strip() or "none"
                        tool_strategies[problem][tool][combo] += 1
            except OSError:
                continue

    # per-problem trimal runs from KB
    trim_by_problem = defaultdict(Counter)
    for exp in experiments:
        problem = exp.get("problem_type", "unknown")
        trim_actions = [
            a for a in exp.get("actions_tried", [])
            if isinstance(a, dict) and a.get("action") == "trim"
        ]
        if trim_actions:
            for a in trim_actions:
                strat = str(a.get("trim_strategy") or a.get("extra_args") or "").strip()
                trim_by_problem[problem][strat or "unknown"] += 1
        elif exp.get("was_trimmed"):
            trim_by_problem[problem]["unknown"] += 1

    print("\n" + "=" * 50)
    print("       TOOL USAGE (by problem type)")
    print("=" * 50)

    problems = _sorted_problems(set(tool_iters) | set(trim_by_problem))
    if not problems:
        print("\nNo tool usage data found (logs/ or knowledge base empty)")
        print("=" * 50)
        return

    tool_order = ["mafft", "muscle", "esm2"]

    for problem in problems:
        counts = tool_iters.get(problem, Counter())
        trims = trim_by_problem.get(problem, Counter())
        if not counts and not trims:
            continue
        tag = "  [reference]" if _is_reference(problem) else ""
        total = sum(counts.values()) + sum(trims.values())
        print(f"\n  ▸ {problem} (n={total} events){tag}")
        max_count = max(list(counts.values()) + list(trims.values()) + [1])
        for tool in tool_order + sorted(t for t in counts if t not in tool_order):
            count = counts.get(tool, 0)
            if count == 0:
                continue
            bar = "█" * max(1, int(count / max_count * 25))
            top = tool_strategies[problem][tool].most_common(1)
            top_str = f"{top[0][0]} ({top[0][1]})" if top else "n/a"
            print(f"    {tool:8} {bar} {count:5} iters | top strategy: {top_str}")
        trim_total = sum(trims.values())
        if trim_total:
            bar = "█" * max(1, int(trim_total / max_count * 25))
            top = trims.most_common(1)
            top_str = f"{top[0][0]} ({top[0][1]})" if top else "n/a"
            print(f"    {'trimal':8} {bar} {trim_total:5} runs  | top strategy: {top_str}")

    print("=" * 50)


def print_esm_gap_stats() -> None:
    """ESM-2 effectiveness by gap band (learned from probe trials)."""
    from rag_retrieve import load_esm_gap_stats, get_esm_threshold

    bands = load_esm_gap_stats()
    thr, learned = get_esm_threshold()

    print("\n" + "=" * 50)
    print("       ESM-2 BY GAP BAND")
    print("=" * 50)

    if not bands:
        print("\nNo esm2 probe trials yet (gap 0.12-0.40 probes will fill this)")
        print("=" * 50)
        return

    def band_lo(b: str) -> float:
        try:
            return float(b.split("-")[0])
        except (ValueError, IndexError):
            return 999.0

    for band in sorted(bands, key=band_lo):
        d = bands[band]
        n = int(d.get("n", 0))
        succ = int(d.get("success", 0))
        avg_delta = (float(d.get("sum_delta", 0)) / n) if n else 0
        rate = (succ / n * 100) if n else 0
        bar = "█" * int(rate / 5)
        print(f"  {band:12} n={n:3}  {bar} {rate:3.0f}%  avg Δgap +{avg_delta:.3f}")

    tag = "learned" if learned else "default (need n>=3 & >=50% per band)"
    print(f"\n  Threshold: gap > {thr:.2f}  [{tag}]")
    print("=" * 50)


def print_esm_post_stats() -> None:
    """ESM post-correction effectiveness by gap band (learning trials)."""
    from rag_retrieve import load_esm_post_stats, get_esm_post_advice

    bands = load_esm_post_stats()

    print("\n" + "=" * 50)
    print("       ESM POST-CORRECTION BY GAP BAND")
    print("=" * 50)

    if not bands:
        print("\nNo esm post-correction trials yet (learning mode fills this)")
        print("=" * 50)
        return

    def band_lo(b: str) -> float:
        try:
            return float(b.split("-")[0])
        except (ValueError, IndexError):
            return 999.0

    for band in sorted(bands, key=band_lo):
        d = bands[band]
        n = int(d.get("n", 0))
        succ = int(d.get("success", 0))
        avg_delta = (float(d.get("sum_delta", 0)) / n) if n else 0
        rate = (succ / n * 100) if n else 0
        # mid-gap for advice label
        try:
            lo = float(band.split("-")[0])
            mid = lo + 0.025
            advice = get_esm_post_advice(mid).upper()
        except (ValueError, IndexError):
            advice = "?"
        bar = "█" * int(rate / 5)
        print(f"  {band:12} n={n:3}  {bar} {rate:3.0f}%  avg Δgap +{avg_delta:.3f}  [{advice}]")

    print("=" * 50)


def print_strategy_table(min_n: int = 3) -> None:
    experiments = load_experiments()
    if not experiments:
        print("No experiments to analyze")
        return

    # (problem, action, strategy, extra) -> stats at experiment level
    stats = defaultdict(lambda: {"n": 0, "success": 0, "exps": []})
    for exp in experiments:
        problem = exp.get("problem_type", "unknown")
        is_success = exp.get("final_outcome") == "success"
        seen = set()
        for a in exp.get("actions_tried", []):
            if not isinstance(a, dict):
                continue
            key = (
                problem,
                a.get("action", ""),
                a.get("strategy", ""),
                a.get("extra_args") or a.get("trim_strategy") or "",
            )
            if key in seen:
                continue
            seen.add(key)
            stats[key]["n"] += 1
            stats[key]["success"] += 1 if is_success else 0
            stats[key]["exps"].append(exp)

    if not stats:
        print("No strategy data found")
        return

    print("\n" + "=" * 50)
    print(f"       STRATEGY TABLE (n>={min_n})")
    print("=" * 50)

    by_problem = defaultdict(list)
    for (problem, action, strategy, extra), data in stats.items():
        if data["n"] < min_n:
            continue
        by_problem[problem].append((action, strategy, extra, data))

    if not by_problem:
        print(f"\nNo strategies with n>={min_n} yet")
        print("=" * 50)
        return

    for problem in _sorted_problems(by_problem):
        rows = sorted(
            by_problem[problem],
            key=lambda r: (r[3]["success"] / r[3]["n"] if r[3]["n"] else 0),
            reverse=True,
        )
        print(f"\n  {problem}")
        for action, strategy, extra, data in rows:
            rate = data["success"] / data["n"] * 100 if data["n"] else 0
            g = _gap_set_stats(data["exps"])
            bar = "█" * int(rate / 10)
            extra_str = extra if extra else "-"
            if g["red_n"]:
                red = f"red {g['red_mean']:.0f}%/{g['red_med']:.0f}%"
            else:
                red = "red —"
            print(
                f"    {action:12} | {strategy:6} | {extra_str:20} "
                f"n={data['n']:3}  {bar} {rate:3.0f}%  {red}"
            )

    print("\n" + "=" * 50)


def _curve_row(prefix: List[Dict], at_n: int) -> str:
    """Cumulative stats row after `at_n` experiments (prefix = first at_n)."""
    n = len(prefix)
    successes = sum(1 for e in prefix if e.get("final_outcome") == "success")
    rate = successes / n
    g = _gap_set_stats(prefix)
    bar = "█" * int(rate * 20)
    if g["red_n"]:
        red = f"init {g['init_mean']:.3f}→{g['final_mean']:.3f}  red mean {g['red_mean']:3.0f}% med {g['red_med']:3.0f}%"
    else:
        red = f"init {g['init_mean']:.3f}→{g['final_mean']:.3f}  red —"
    return f"    at n={at_n:3}:  {bar} {rate*100:3.0f}%  {red}"


def _cumulative_checkpoints(n: int) -> List[int]:
    """~4 checkpoints per problem type: quartiles of n, always include n."""
    if n <= 0:
        return []
    if n < 3:
        return list(range(1, n + 1))
    step = max(1, math.ceil(n / 4))
    pts = list(range(step, n, step))
    if not pts or pts[-1] != n:
        pts.append(n)
    return pts


def print_learning_curve(buckets: int = 10) -> None:
    experiments = load_experiments()
    if not experiments:
        print("No experiments to show")
        return

    dated = [e for e in experiments if e.get("timestamp")]
    dated.sort(key=lambda e: e.get("timestamp", ""))
    if not dated:
        print("No timestamped experiments")
        return

    by_problem = defaultdict(list)
    for e in dated:
        by_problem[e.get("problem_type", "unknown")].append(e)

    print("\n" + "=" * 50)
    print("       LEARNING CURVE (cumulative as analyses were added)")
    print("=" * 50)

    for problem in _sorted_problems(by_problem):
        group = by_problem[problem]
        n = len(group)
        tag = "  [reference — any strategy works]" if _is_reference(problem) else ""
        print(f"\n  ▸ {problem} (n={n}){tag}")

        if n < 3:
            rate = sum(1 for e in group if e.get("final_outcome") == "success") / n * 100
            g = _gap_set_stats(group)
            red = f"red mean {g['red_mean']:.0f}% med {g['red_med']:.0f}%" if g["red_n"] else "red —"
            print(f"    (too few points) success {rate:.0f}%  {red}")
            continue

        pts = _cumulative_checkpoints(n)
        for at_n in pts:
            print(_curve_row(group[:at_n], at_n))

        first_g = _gap_set_stats(group[: pts[0]])
        last_g = _gap_set_stats(group)
        if first_g["red_n"] and last_g["red_n"]:
            delta = last_g["red_mean"] - first_g["red_mean"]
            arrow = "↑" if delta > 0.5 else ("↓" if delta < -0.5 else "→")
            print(
                f"    Growth: red mean {first_g['red_mean']:.0f}% → {last_g['red_mean']:.0f}%  "
                f"({delta:+.0f} pts over +{n - pts[0]} analyses) {arrow}"
            )

    print("\n" + "=" * 50)


def print_strategy_performance() -> None:
    experiments = load_experiments()

    if not experiments:
        print("No experiments to analyze")
        return

    by_problem = defaultdict(lambda: defaultdict(lambda: {"success": 0, "total": 0}))

    for exp in experiments:
        problem = exp.get("problem_type", "unknown")
        actions = exp.get("actions_tried", [])
        outcome = exp.get("final_outcome", "")

        for action in actions:
            key = f"{action.get('action')} | {action.get('strategy')} | {action.get('extra_args') or 'none'}"
            by_problem[problem][key]["total"] += 1
            if outcome in ("success", "partial_improvement"):
                by_problem[problem][key]["success"] += 1

    print("\n" + "=" * 50)
    print("       STRATEGY PERFORMANCE (by problem type)")
    print("=" * 50)

    for problem in _sorted_problems(by_problem):
        strategies = by_problem[problem]
        tag = "  [reference]" if _is_reference(problem) else ""
        print(f"\n  ▸ {problem}{tag}")
        sorted_strategies = sorted(
            strategies.items(),
            key=lambda x: x[1]["success"] / x[1]["total"] if x[1]["total"] > 0 else 0,
            reverse=True,
        )
        for strategy, data in sorted_strategies[:10]:
            rate = (data["success"] / data["total"] * 100) if data["total"] > 0 else 0
            bar = "█" * int(rate / 10)
            print(f"    {strategy}")
            print(f"       {bar} {rate:.0f}% ({data['success']}/{data['total']})")

    print("\n" + "=" * 50)


def print_learning_timeline() -> None:
    experiments = load_experiments()

    if not experiments:
        print("No experiments to show")
        return

    experiments.sort(key=lambda x: x.get("timestamp", ""))

    print("\n" + "=" * 50)
    print("       LEARNING TIMELINE")
    print("=" * 50)

    for exp in experiments:
        ts = exp.get("timestamp", "")[:19]
        problem = exp.get("problem_type", "?")[:15]
        outcome = exp.get("final_outcome", "?")
        gap_in = exp.get("initial_gap_fraction", 0)
        gap_out = exp.get("final_gap_fraction", 0)

        outcome_symbol = "✓" if outcome == "success" else "~" if outcome == "partial_improvement" else "✗"

        print(f"\n{ts} | {problem:15} | {outcome_symbol} | gap: {gap_in:.2f} -> {gap_out:.2f}")

        actions = exp.get("actions_tried", [])
        for a in actions:
            action = a.get("action", "")
            strategy = a.get("strategy", "")
            extra = a.get("extra_args", "")
            print(f"    -> {action} {strategy} {extra}")

    print("\n" + "=" * 50)


def generate_text_report() -> str:
    experiments = load_experiments()
    feedback = load_feedback()

    report = "MSA AI LEARNING REPORT\n"
    report += "=" * 40 + "\n\n"

    report += f"Total Experiments: {len(experiments)}\n"
    report += f"Total Feedback: {len(feedback)}\n\n"

    by_problem = defaultdict(int)
    for exp in experiments:
        by_problem[exp.get("problem_type", "unknown")] += 1

    report += "Problem Types:\n"
    for pt, count in sorted(by_problem.items()):
        report += f"  - {pt}: {count}\n"

    if experiments:
        successes = sum(1 for e in experiments if e.get("final_outcome") == "success")
        report += f"\nSuccess Rate: {successes}/{len(experiments)} ({successes/len(experiments)*100:.1f}%)\n"

    return report


def export_text_report(filename: str = "learning_report.txt") -> None:
    report = generate_text_report()
    with open(filename, "w") as f:
        f.write(report)
    print(f"Report saved to: {filename}")