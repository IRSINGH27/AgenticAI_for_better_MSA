import json
import os
import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from collections import defaultdict
import pathlib

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
KNOWLEDGE_DIR = pathlib.Path(os.environ.get("MSA_KNOWLEDGE_DIR") or (SCRIPT_DIR.parent / "knowledge"))
TOON_FILE = "experiments.toon"
JSONL_FILE = "experiments.jsonl"


def load_experiments() -> List[Dict]:
    from rag_retrieve import load_knowledge
    return load_knowledge()


def calculate_confidence_weight(experiments: List[Dict], current_time: datetime) -> Dict[str, float]:
    weights = {}

    for exp in experiments:
        exp_id = exp.get("experiment_id")
        timestamp_str = exp.get("timestamp", "")

        confidence_weight = 1.0
        outcome_weight = 1.0

        decisions = exp.get("actions_tried", [])
        for decision in decisions:
            conf = decision.get("confidence", "medium").lower()
            if conf == "high":
                confidence_weight += 0.3
            elif conf == "low":
                confidence_weight -= 0.1

            outcome = decision.get("outcome", "")
            if outcome == "success":
                outcome_weight += 0.5
            elif outcome == "partial_improvement":
                outcome_weight += 0.2

        if timestamp_str:
            try:
                exp_time = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                if isinstance(exp_time, datetime):
                    days_ago = (current_time - exp_time.replace(tzinfo=None)).days
                    time_decay = max(0.5, 1.0 - (days_ago / 365))
                    confidence_weight *= time_decay
            except:
                pass

        weights[exp_id] = round(confidence_weight, 2)

    return weights


def get_weighted_recommendations(experiments: List[Dict], problem_type: str, current_time: datetime) -> Dict:
    weights = calculate_confidence_weight(experiments, current_time)

    strategy_scores = defaultdict(lambda: {"weighted_score": 0, "count": 0, "successes": 0})

    for exp in experiments:
        if exp.get("problem_type") != problem_type:
            continue

        exp_weight = weights.get(exp.get("experiment_id"), 1.0)
        exp_outcome = exp.get("final_outcome", "")

        actions = exp.get("actions_tried", [])
        for action in actions:
            strategy = f"{action.get('action')}:{action.get('strategy')}:{action.get('extra_args')}"
            outcome = action.get("outcome", "")

            if outcome == "success":
                strategy_scores[strategy]["weighted_score"] += exp_weight * 1.0
                strategy_scores[strategy]["successes"] += 1
            elif outcome == "partial_improvement":
                strategy_scores[strategy]["weighted_score"] += exp_weight * 0.5
            elif outcome == "no_change":
                strategy_scores[strategy]["weighted_score"] -= exp_weight * 0.3

            strategy_scores[strategy]["count"] += 1

    recommendations = {}
    for strategy, data in strategy_scores.items():
        if data["count"] >= 1:
            avg_score = data["weighted_score"] / data["count"]
            success_rate = data["successes"] / data["count"]
            recommendations[strategy] = {
                "weighted_score": round(avg_score, 2),
                "success_rate": round(success_rate, 2),
                "total_runs": data["count"],
                "successes": data["successes"]
            }

    return recommendations


def _fingerprint_key(exp: Dict) -> str:
    return str(exp.get("fingerprint") or exp.get("sequence_signature") or exp.get("experiment_id") or "")


def _make_folds(experiments: List[Dict], k: int, seed: int = 42) -> List[List[Dict]]:
    import random
    rng = random.Random(seed)

    # Group near-duplicates (same fingerprint) so they never cross folds
    groups: Dict[str, List[Dict]] = defaultdict(list)
    for exp in experiments:
        groups[_fingerprint_key(exp)].append(exp)

    # Stratify groups by problem_type when possible
    by_problem: Dict[str, List[List[Dict]]] = defaultdict(list)
    for group in groups.values():
        problem = group[0].get("problem_type", "unknown")
        by_problem[problem].append(group)

    ordered_groups: List[List[Dict]] = []
    for problem in sorted(by_problem):
        problem_groups = by_problem[problem]
        rng.shuffle(problem_groups)
        ordered_groups.extend(problem_groups)

    folds: List[List[Dict]] = [[] for _ in range(k)]
    fold_sizes = [0] * k
    for group in ordered_groups:
        lightest = min(range(k), key=lambda i: fold_sizes[i])
        folds[lightest].extend(group)
        fold_sizes[lightest] += len(group)

    return [f for f in folds if f]


def _strategy_key(action: Dict) -> str:
    return f"{action.get('action', '')}:{action.get('strategy', '')}:{action.get('extra_args') or ''}"


def _best_strategy_from_train(train: List[Dict], problem_type: str, min_n: int = 3) -> Optional[str]:
    stats = defaultdict(lambda: {"n": 0, "successes": 0, "gap_delta": 0.0})
    for exp in train:
        if exp.get("problem_type") != problem_type:
            continue
        delta = float(exp.get("initial_gap_fraction", 0) or 0) - float(exp.get("final_gap_fraction", 0) or 0)
        is_success = exp.get("final_outcome") == "success"
        seen = set()
        for action in exp.get("actions_tried", []):
            if not isinstance(action, dict):
                continue
            key = _strategy_key(action)
            if key in seen:
                continue
            seen.add(key)
            stats[key]["n"] += 1
            stats[key]["successes"] += 1 if is_success else 0
            stats[key]["gap_delta"] += delta

    best_key = None
    best_rate = -1.0
    for key, data in stats.items():
        if data["n"] < min_n:
            continue
        rate = data["successes"] / data["n"]
        if rate > best_rate or (rate == best_rate and data["n"] > stats.get(best_key, {}).get("n", 0)):
            best_rate = rate
            best_key = key
    return best_key


def _predicted_gap_delta(train: List[Dict], problem_type: str, strategy_key: Optional[str]) -> float:
    deltas = []
    fallback = []
    for exp in train:
        if exp.get("problem_type") != problem_type:
            continue
        delta = float(exp.get("initial_gap_fraction", 0) or 0) - float(exp.get("final_gap_fraction", 0) or 0)
        fallback.append(delta)
        if strategy_key is None:
            continue
        keys = {_strategy_key(a) for a in exp.get("actions_tried", []) if isinstance(a, dict)}
        if strategy_key in keys:
            deltas.append(delta)
    pool = deltas or fallback
    if not pool:
        return 0.0
    return sum(pool) / len(pool)


def cross_validate(experiments: List[Dict], k: int = 5) -> Dict:
    """Stratified, fingerprint-grouped k-fold cross-validation.

    Compares two predictors against final_outcome == "success":
      - baseline: problem-type success rate in train > 0.5
      - strategy-aware: best (action:strategy:extra_args) with n>=3 in train
    Also scores probability calibration (Brier) and gap-improvement MAE.
    """
    if len(experiments) < k * 2:
        k = max(2, len(experiments) // 2)
    if len(experiments) < 4:
        return {"message": "Not enough experiments for cross-validation", "results": [], "accuracy": 0.0}

    folds = _make_folds(experiments, k)
    if len(folds) < 2:
        return {"message": "Not enough distinct groups for cross-validation", "results": [], "accuracy": 0.0}

    results = []
    correct_base = 0
    correct_strat = 0
    brier_sum = 0.0
    gap_mae_sum = 0.0
    scored = 0

    for i in range(len(folds)):
        test = folds[i]
        train = [e for j, f in enumerate(folds) if j != i for e in f]
        if not train or not test:
            continue

        for exp in test:
            problem = exp.get("problem_type", "unknown")
            actual_success = exp.get("final_outcome") == "success"
            actual_gap_delta = float(exp.get("initial_gap_fraction", 0) or 0) - float(exp.get("final_gap_fraction", 0) or 0)

            same_problem = [e for e in train if e.get("problem_type") == problem]
            if same_problem:
                base_rate = sum(1 for e in same_problem if e.get("final_outcome") == "success") / len(same_problem)
            else:
                all_rate = sum(1 for e in train if e.get("final_outcome") == "success") / len(train)
                base_rate = all_rate

            strat_key = _best_strategy_from_train(train, problem)
            if strat_key is not None:
                # success rate of that strategy on same-problem train experiments
                strat_n = 0
                strat_s = 0
                for e in train:
                    if e.get("problem_type") != problem:
                        continue
                    keys = {_strategy_key(a) for a in e.get("actions_tried", []) if isinstance(a, dict)}
                    if strat_key in keys:
                        strat_n += 1
                        strat_s += 1 if e.get("final_outcome") == "success" else 0
                strat_rate = strat_s / strat_n if strat_n else base_rate
            else:
                strat_rate = base_rate

            pred_gap = _predicted_gap_delta(train, problem, strat_key)

            base_pred = base_rate > 0.5
            strat_pred = strat_rate > 0.5
            correct_base += 1 if base_pred == actual_success else 0
            correct_strat += 1 if strat_pred == actual_success else 0
            brier_sum += (strat_rate - (1.0 if actual_success else 0.0)) ** 2
            gap_mae_sum += abs(pred_gap - actual_gap_delta)
            scored += 1

            results.append({
                "experiment_id": exp.get("experiment_id"),
                "problem_type": problem,
                "baseline_rate": round(base_rate, 2),
                "strategy_rate": round(strat_rate, 2),
                "recommended_strategy": strat_key,
                "predicted_gap_delta": round(pred_gap, 4),
                "actual_gap_delta": round(actual_gap_delta, 4),
                "actual_outcome": exp.get("final_outcome", ""),
            })

    if scored == 0:
        return {"message": "No experiments scored", "results": [], "accuracy": 0.0}

    accuracy = correct_strat / scored
    baseline_accuracy = correct_base / scored
    per_problem = defaultdict(lambda: {"n": 0, "correct": 0, "gap_delta_sum": 0.0})
    for r in results:
        p = per_problem[r["problem_type"]]
        p["n"] += 1
        p["correct"] += 1 if (r["strategy_rate"] > 0.5) == (r["actual_outcome"] == "success") else 0
        p["gap_delta_sum"] += r["actual_gap_delta"]

    return {
        "accuracy": round(accuracy, 3),
        "baseline_accuracy": round(baseline_accuracy, 3),
        "lift": round(accuracy - baseline_accuracy, 3),
        "brier_score": round(brier_sum / scored, 3),
        "gap_mae": round(gap_mae_sum / scored, 4),
        "folds": len(folds),
        "train_size": len(experiments) - scored,
        "test_size": scored,
        "per_problem": {
            p: {
                "n": d["n"],
                "accuracy": round(d["correct"] / d["n"], 3) if d["n"] else 0.0,
                "avg_gap_delta": round(d["gap_delta_sum"] / d["n"], 4) if d["n"] else 0.0,
            }
            for p, d in sorted(per_problem.items())
        },
        "results": results,
    }


def get_learning_progress() -> Dict:
    experiments = load_experiments()
    if not experiments:
        return {"message": "No experiments yet"}

    by_problem = defaultdict(lambda: {"total": 0, "success": 0})
    by_month = defaultdict(int)
    success_trend = []

    for exp in experiments:
        problem = exp.get("problem_type", "unknown")
        by_problem[problem]["total"] += 1
        if exp.get("final_outcome") == "success":
            by_problem[problem]["success"] += 1

        timestamp = exp.get("timestamp", "")
        if timestamp:
            try:
                month = timestamp[:7]
                by_month[month] += 1
            except:
                pass

    sorted_months = sorted(by_month.items())
    for i, (month, count) in enumerate(sorted_months):
        if i > 0:
            change = count - sorted_months[i-1][1]
            success_trend.append({"month": month, "experiments": count, "change": change})

    progress = {
        "total_experiments": len(experiments),
        "by_problem_type": dict(by_problem),
        "success_rates": {
            k: round(v["success"] / v["total"], 2) if v["total"] > 0 else 0
            for k, v in by_problem.items()
        },
        "monthly_counts": dict(sorted_months),
        "trend": success_trend[-6:]
    }

    return progress