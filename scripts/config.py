import pathlib

PROJECT_ROOT = pathlib.Path(__file__).parent.parent
SCRIPTS_DIR = pathlib.Path(__file__).parent

CONDA_ENV = "seqTools"

KNOWLEDGE_DIR = PROJECT_ROOT / "knowledge"
LOGS_DIR = PROJECT_ROOT / "logs"
DATA_DIR = PROJECT_ROOT / "data"
FAMILIES_DIR = DATA_DIR / "pfam" / "families"
MODELS_DIR = PROJECT_ROOT / "models"
