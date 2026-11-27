import os
import pathlib

PROJECT_ROOT = pathlib.Path(__file__).parent.parent
SCRIPTS_DIR = pathlib.Path(__file__).parent

CONDA_ENV = "seqTools"

# Env-var overrides let benchmarks isolate their KB/logs without touching defaults
KNOWLEDGE_DIR = pathlib.Path(os.environ.get("MSA_KNOWLEDGE_DIR") or (PROJECT_ROOT / "knowledge"))
LOGS_DIR = pathlib.Path(os.environ.get("MSA_LOGS_DIR") or (PROJECT_ROOT / "logs"))
DATA_DIR = PROJECT_ROOT / "data"
FAMILIES_DIR = DATA_DIR / "pfam" / "families"
MODELS_DIR = PROJECT_ROOT / "models"
