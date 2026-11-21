from typing import Optional


class ConvergenceTracker:
    def __init__(self, tolerance: float, patience: int):
        self.tolerance = tolerance
        self.patience = patience
        self.previous_metrics: Optional[dict] = None
        self.stable_count = 0
        self.iteration = 0
        self.best_gap = float('inf')
        self.strategies_tried = 0

    def check(self, metrics: dict) -> tuple[bool, dict]:
        self.iteration += 1

        if self.previous_metrics is None:
            self.previous_metrics = metrics.copy()
            self.best_gap = metrics.get("gap_fraction", 1.0)
            return False, {"status": "init", "instability": 1.0}

        gap = metrics.get("gap_fraction", 0)
        gap_diff = abs(gap - self.previous_metrics.get("gap_fraction", 0))
        len_diff = abs(metrics.get("alignment_length", 0) - self.previous_metrics.get("alignment_length", 0))
        len_scale = max(self.previous_metrics.get("alignment_length", 1), 1)

        instability = gap_diff + (len_diff / len_scale)

        if gap < self.best_gap - self.tolerance:
            self.best_gap = gap
            self.stable_count = 0
        elif instability <= self.tolerance:
            self.stable_count += 1
        else:
            self.stable_count = 0

        self.previous_metrics = metrics.copy()

        converged = self.stable_count >= self.patience

        if converged:
            status = "converged"
        else:
            status = "running"

        return converged, {"status": status, "instability": float(instability)}
