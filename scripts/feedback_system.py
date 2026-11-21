import json
import os
from datetime import datetime
from typing import Dict, List, Optional
import pathlib

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
KNOWLEDGE_DIR = SCRIPT_DIR.parent / "knowledge"
FEEDBACK_FILE = "user_feedback.json"


def load_feedback() -> List[Dict]:
    path = os.path.join(KNOWLEDGE_DIR, FEEDBACK_FILE)
    if not os.path.exists(path):
        return []

    with open(path, "r") as f:
        return json.load(f)


def save_feedback(feedback: List[Dict]) -> None:
    os.makedirs(KNOWLEDGE_DIR, exist_ok=True)
    path = os.path.join(KNOWLEDGE_DIR, FEEDBACK_FILE)
    with open(path, "w") as f:
        json.dump(feedback, f, indent=2)


def add_feedback(experiment_id: str, rating: int, comment: str = "") -> bool:
    if rating < 1 or rating > 5:
        print("Rating must be between 1-5")
        return False

    feedback = load_feedback()

    feedback.append({
        "experiment_id": experiment_id,
        "rating": rating,
        "comment": comment,
        "timestamp": datetime.now().isoformat()
    })

    save_feedback(feedback)
    print(f"Feedback saved for experiment {experiment_id}")
    return True


def get_experiment_feedback(experiment_id: str) -> Optional[Dict]:
    feedback = load_feedback()
    for f in feedback:
        if f.get("experiment_id") == experiment_id:
            return f
    return None


def update_strategy_weights() -> Dict:
    feedback = load_feedback()
    from rag_retrieve import load_knowledge
    experiments = load_knowledge()

    if not experiments:
        return {"message": "No experiments found"}

    strategy_scores = {}

    for exp in experiments:
        exp_id = exp.get("experiment_id")
        exp_feedback = get_experiment_feedback(exp_id)

        rating = exp_feedback.get("rating", 3) if exp_feedback else 3

        actions = exp.get("actions_tried", [])
        for action in actions:
            strategy = f"{action.get('action')}:{action.get('strategy')}:{action.get('extra_args')}"

            if strategy not in strategy_scores:
                strategy_scores[strategy] = {"total_rating": 0, "count": 0}

            strategy_scores[strategy]["total_rating"] += rating
            strategy_scores[strategy]["count"] += 1

    adjusted_strategies = {}
    for strategy, data in strategy_scores.items():
        avg_rating = data["total_rating"] / data["count"] if data["count"] > 0 else 3.0
        adjusted_strategies[strategy] = {
            "avg_user_rating": round(avg_rating, 2),
            "times_used": data["count"],
            "weight": round((avg_rating - 1) / 4, 2)
        }

    return adjusted_strategies


def get_feedback_stats() -> Dict:
    feedback = load_feedback()

    if not feedback:
        return {"message": "No feedback yet"}

    ratings = [f.get("rating", 0) for f in feedback]
    avg_rating = sum(ratings) / len(ratings) if ratings else 0

    return {
        "total_feedback": len(feedback),
        "average_rating": round(avg_rating, 2),
        "rating_distribution": {
            "1": ratings.count(1),
            "2": ratings.count(2),
            "3": ratings.count(3),
            "4": ratings.count(4),
            "5": ratings.count(5)
        }
    }       


def interactive_feedback(experiment_id: str) -> None:
    print(f"\nProvide feedback for experiment: {experiment_id}")
    print("Rate the outcome (1-5):")
    print("  1 - Very Poor")
    print("  2 - Poor")
    print("  3 - Average")
    print("  4 - Good")
    print("  5 - Excellent")

    try:
        rating = int(input("Rating: "))
    except ValueError:
        print("Invalid rating")
        return

    comment = input("Optional comment: ")

    add_feedback(experiment_id, rating, comment)
    print("Thank you for your feedback!")