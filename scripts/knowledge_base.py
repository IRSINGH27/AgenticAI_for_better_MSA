import json
import os
from typing import Dict, List, Optional

KNOWLEDGE_DIR = "knowledge"
EXTERNAL_FILE = "external_knowledge.json"


MAFFT_DOCS = {
    "strategies": {
        "--auto": "Automatically selects best method based on data size. Good default choice.",
        "--localpair": "Local alignment (L-INS-i). Best for <200 sequences with diverse lengths. Slow but accurate.",
        "--globalpair": "Global alignment (G-INS-i). Best for similar sequences with similar lengths.",
        "--genafpair": "Generalized affine (E-INS-i). Good for sequences with multiple conserved domains.",
        "--retree": "Fast tree-based method. Quick for large datasets.",
        "--fast": "Fast heuristics. Use for quick initial alignment."
    },
    "parameters": {
        "--maxiterate <n>": "Max iterative refinements. Use at least 1000 for accurate alignment; values below 1000 are raised to 1000 by the pipeline.",
        "--op <float>": "Gap opening penalty. Higher = fewer gaps. (default: 1.53)",
        "--ep <float>": "Gap extension penalty. Higher = shorter gaps. (default: 0.0)",
        "--op <float> --ep <float>": "Combined: --op 2.0 --ep 0.5 for stricter gap control",
        "--nuc": "Force nucleotide mode. Use for DNA/RNA sequences.",
        "--amino": "Force amino acid mode. Use for protein sequences.",
        "--thread <n>": "Number of CPU threads. Use -1 for all available.",
        "--reorder": "Reorder sequences by similarity in output.",
        "--treeout": "Output guide tree used for alignment."
    },
    "best_practices": {
        "high_gap_fraction": "Try --localpair with --maxiterate 1000 or use trimal to trim.",
        "diverse_sequences": "Use --localpair or --genafpair for sequences with multiple domains.",
        "similar_sequences": "Use --globalpair for closely related sequences.",
        "large_dataset": "Use --retree or --fast for quick alignment, then refine.",
        "nucleotide_vs_protein": "Use --nuc for DNA/RNA, --amino for proteins."
    }
}

TRIMAL_DOCS = {
    "strategies": {
        "-gappyout": "Automated selection from gap distribution. Gentle, recommended default.",
        "-automated1": "Heuristic auto selection from similarity statistics. Gentle."
    },
    "parameters": {
        "-gt <float>": "Gap threshold (0.0-1.0). Alternative to strategy flags.",
        "-cons <float>": "Minimum % for consensus column.",
        "-seqoverlap <int>": "Minimum percentage of good positions a sequence must keep (0-100).",
        "-htmlout <file>": "Generate HTML visualization.",
        "-fasta": "Force FASTA output format."
    },
    "best_practices": {
        "high_gap_fraction": "Use -gappyout (gentle). Never -strict/-strictplus/-nogaps — they delete indel/gap columns.",
        "protein_alignment": "Use -gappyout for protein sequences (preserves indels).",
        "nucleotide": "Use -gappyout for nucleotide to preserve more alignment.",
        "conserved_domains": "Use -automated1; do not use -nogaps (loses indels)."
    }
}


def load_external_knowledge() -> Dict:
    path = os.path.join(KNOWLEDGE_DIR, EXTERNAL_FILE)
    if not os.path.exists(path):
        return {}

    with open(path, "r") as f:
        return json.load(f)


def save_external_knowledge(data: Dict) -> None:
    os.makedirs(KNOWLEDGE_DIR, exist_ok=True)
    path = os.path.join(KNOWLEDGE_DIR, EXTERNAL_FILE)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def add_custom_knowledge(entry: Dict) -> None:
    data = load_external_knowledge()
    if "custom" not in data:
        data["custom"] = []

    entry["added_at"] = str(datetime.now().isoformat())
    data["custom"].append(entry)

    save_external_knowledge(data)


def get_mafft_help(problem_type: str) -> str:
    help_text = "=== MAFFT DOCUMENTATION ===\n\n"

    help_text += "STRATEGIES:\n"
    for strategy, desc in MAFFT_DOCS["strategies"].items():
        help_text += f"  {strategy}: {desc}\n"

    help_text += "\nPARAMETERS:\n"
    for param, desc in MAFFT_DOCS["parameters"].items():
        help_text += f"  {param}: {desc}\n"

    if problem_type in MAFFT_DOCS["best_practices"]:
        help_text += f"\nFOR {problem_type.upper()}:\n"
        help_text += f"  {MAFFT_DOCS['best_practices'][problem_type]}\n"

    return help_text


def get_trimal_help(problem_type: str = None) -> str:
    help_text = "=== TRIMAL DOCUMENTATION ===\n\n"

    help_text += "STRATEGIES:\n"
    for strategy, desc in TRIMAL_DOCS["strategies"].items():
        help_text += f"  {strategy}: {desc}\n"

    help_text += "\nPARAMETERS:\n"
    for param, desc in TRIMAL_DOCS["parameters"].items():
        help_text += f"  {param}: {desc}\n"

    if problem_type and problem_type in TRIMAL_DOCS["best_practices"]:
        help_text += f"\nFOR {problem_type.upper()}:\n"
        help_text += f"  {TRIMAL_DOCS['best_practices'][problem_type]}\n"

    return help_text


def get_external_context(problem_type: str, include_mafft: bool = True, include_trimal: bool = True) -> str:
    context = ""

    if include_mafft:
        context += get_mafft_help(problem_type) + "\n\n"

    if include_trimal:
        context += get_trimal_help(problem_type)

    return context


from datetime import datetime