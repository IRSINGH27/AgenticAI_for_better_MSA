import os
import json
from datetime import datetime
from typing import Any


class ExperimentLogger:
    def __init__(self):
        self.entries = []
        self.logs_dir = "logs"
        self._ensure_logs_dir()

    def _ensure_logs_dir(self):
        os.makedirs(self.logs_dir, exist_ok=True)

    def log_step(self, entry: dict):
        entry_copy = entry.copy()
        entry_copy["timestamp"] = datetime.utcnow().isoformat()
        self.entries.append(entry_copy)

    def save(self, experiment_id: str):
        output_path = os.path.join(self.logs_dir, f"{experiment_id}.jsonl")
        with open(output_path, "a") as f:
            for entry in self.entries:
                f.write(json.dumps(entry) + "\n")
        self.entries.clear()