import json
import os
import shutil
from datetime import datetime
from typing import Dict, List, Optional
import zipfile
import pathlib

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
KNOWLEDGE_DIR = SCRIPT_DIR.parent / "knowledge"


def export_knowledge(output_path: Optional[str] = None) -> str:
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"knowledge_export_{timestamp}.zip"

    knowledge_files = [
        "experiments.jsonl",
        "strategy_stats.json",
        "user_feedback.json",
        "external_knowledge.json"
    ]

    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for filename in knowledge_files:
            filepath = os.path.join(KNOWLEDGE_DIR, filename)
            if os.path.exists(filepath):
                zipf.write(filepath, filename)
                print(f"Added: {filename}")

    print(f"\nKnowledge base exported to: {output_path}")
    return output_path


def import_knowledge(input_path: str, merge: bool = True) -> Dict:
    if not os.path.exists(input_path):
        return {"error": f"File not found: {input_path}"}

    temp_dir = "/tmp/knowledge_import_temp"
    os.makedirs(temp_dir, exist_ok=True)

    try:
        with zipfile.ZipFile(input_path, "r") as zipf:
            zipf.extractall(temp_dir)
    except:
        return {"error": "Invalid ZIP file"}

    imported = {"experiments": 0, "feedback": 0, "stats": 0, "external": 0}

    if merge:
        for filename in os.listdir(temp_dir):
            source = os.path.join(temp_dir, filename)
            dest = os.path.join(KNOWLEDGE_DIR, filename)

            if filename == "experiments.jsonl":
                if os.path.exists(dest):
                    with open(dest, "r") as f:
                        existing = [json.loads(l) for l in f if l.strip()]

                    with open(source, "r") as f:
                        new_exp = [json.loads(l) for l in f if l.strip()]

                    existing_ids = {e.get("experiment_id") for e in existing}
                    for exp in new_exp:
                        if exp.get("experiment_id") not in existing_ids:
                            existing.append(exp)
                            imported["experiments"] += 1

                    with open(dest, "w") as f:
                        for exp in existing:
                            f.write(json.dumps(exp) + "\n")
                else:
                    shutil.copy(source, dest)
                    with open(dest, "r") as f:
                        imported["experiments"] = sum(1 for l in f if l.strip())

            elif filename == "user_feedback.json":
                if os.path.exists(dest):
                    with open(dest, "r") as f:
                        existing = json.load(f)
                    with open(source, "r") as f:
                        new = json.load(f)
                    existing.extend(new)
                    imported["feedback"] = len(existing)
                    with open(dest, "w") as f:
                        json.dump(existing, f, indent=2)
                else:
                    shutil.copy(source, dest)
                    with open(dest, "r") as f:
                        imported["feedback"] = len(json.load(f))

            elif filename in ["strategy_stats.json", "external_knowledge.json"]:
                if os.path.exists(dest):
                    with open(dest, "r") as f:
                        existing = json.load(f)
                    with open(source, "r") as f:
                        new = json.load(f)
                    existing.update(new)
                    with open(dest, "w") as f:
                        json.dump(existing, f, indent=2)
                else:
                    shutil.copy(source, dest)
                imported["stats"] += 1

    shutil.rmtree(temp_dir)

    print(f"\nKnowledge base imported successfully!")
    print(f"  Experiments: {imported['experiments']}")
    print(f"  Feedback: {imported['feedback']}")
    print(f"  Stats: {imported['stats']}")

    return imported


def clean_knowledge(keep_recent: int = 10) -> Dict:
    path = os.path.join(KNOWLEDGE_DIR, "experiments.jsonl")
    if not os.path.exists(path):
        return {"message": "No experiments to clean"}

    with open(path, "r") as f:
        experiments = [json.loads(l) for l in f if l.strip()]

    experiments.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

    kept = experiments[:keep_recent]
    removed = len(experiments) - len(kept)

    with open(path, "w") as f:
        for exp in kept:
            f.write(json.dumps(exp) + "\n")

    return {
        "kept": len(kept),
        "removed": removed,
        "message": f"Kept {len(kept)} most recent experiments"
    }