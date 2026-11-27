import json
import os
import subprocess
from datetime import datetime
import pathlib

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
KNOWLEDGE_DIR = pathlib.Path(os.environ.get("MSA_KNOWLEDGE_DIR") or (SCRIPT_DIR.parent / "knowledge"))
JSONL_GZ_FILE = "experiments.jsonl.gz"
INDEX_SUFFIX = "tfidf_index.npz + tfidf_index_vocab.json"


def _get_problem_type(gap_fraction: float, instability: float) -> str:
    if instability > 0.1:
        return "unstable"
    elif gap_fraction > 0.2:
        return "high_gap_fraction"
    elif gap_fraction > 0.1:
        return "moderate_gap_fraction"
    else:
        return "good_quality"


def _classify_outcome(initial_gap: float, final_gap: float, was_trimmed: bool) -> str:
    if final_gap < 0.1:
        return "success"
    elif initial_gap > final_gap:
        return "partial_improvement"
    else:
        return "no_improvement"


def _compute_sequence_signature(fasta_text: str, k: int = 4) -> set:
    seq = fasta_text.replace(">", "").replace("\n", "").replace(" ", "").upper()
    return set(seq[i:i+k] for i in range(len(seq) - k + 1))


def _load_existing_records():
    from rag_retrieve import load_knowledge
    return load_knowledge()


def _append_record(record: dict) -> None:
    gz_path = KNOWLEDGE_DIR / JSONL_GZ_FILE
    data = (json.dumps(record) + "\n").encode("utf-8")
    with open(gz_path, "ab") as f:
        subprocess.run(["gzip", "-c"], input=data, stdout=f, check=True)


def _sanitize_metrics(metrics: dict) -> dict:
    sanitized = dict(metrics)
    if "hill_problematic_columns" in sanitized:
        cols = sanitized["hill_problematic_columns"]
        if isinstance(cols, list):
            sanitized["hill_problematic_columns"] = cols[:10]
    return sanitized


def _is_duplicate(fasta_text: str, problem_type: str, threshold: float = 0.7) -> bool:
    from tfidf_sparse import load_index, compute_query_vector, query_sparse

    matrix, vocab, exp_ids = load_index()
    if matrix.shape[0] == 0 or not vocab:
        return False

    query_vec = compute_query_vector(fasta_text, vocab)
    results = query_sparse(query_vec, matrix, exp_ids, top_k=1)

    if not results:
        return False

    _, similarity = results[0]
    return similarity >= threshold


def store_experiment(
    experiment_id: str,
    fasta_text: str,
    metrics_history: list,
    decisions: list,
    final_metrics: dict,
    was_trimmed: bool,
    final_alignment: str,
    esm_post_applied: bool = False,
    esm_gap_before: float = 0.0,
    esm_gap_after: float = 0.0,
) -> None:
    from rag_retrieve import update_strategy_stats
    from embeddings import compute_tfidf_embedding, get_sequence_fingerprint
    from tfidf_sparse import (
        build_sparse_index, save_index, load_index, add_to_index
    )

    KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)

    num_sequences = fasta_text.count(">")
    total_chars = len(fasta_text.replace("\n", "").replace(">", ""))
    avg_seq_len = total_chars / max(num_sequences, 1)
    sequence_signature = _compute_sequence_signature(fasta_text)
    tfidf_embedding = compute_tfidf_embedding(fasta_text)
    fingerprint = get_sequence_fingerprint(fasta_text)

    initial_gap = metrics_history[0]["gap_fraction"] if metrics_history else 0
    final_gap = final_metrics.get("gap_fraction", 0)
    problem_type = _get_problem_type(initial_gap, 0)

    if _is_duplicate(fasta_text, problem_type):
        print(f"  -> Skipped: similar experiment already in knowledge base")
        return

    actions_tried = []
    for i, decision in enumerate(decisions):
        metrics = metrics_history[i] if i < len(metrics_history) else {}
        outcome = "no_change"
        if i == len(decisions) - 1:
            outcome = _classify_outcome(initial_gap, final_gap, was_trimmed)

        action_entry = {
            "iteration": i + 1,
            "action": decision.get("action", ""),
            "strategy": decision.get("msa_mode", ""),
            "extra_args": decision.get("extra_args", ""),
            "tool": decision.get("_tool_used") or decision.get("tool", ""),
            "gap_fraction": round(metrics.get("gap_fraction", 0), 4),
            "outcome": outcome
        }
        if was_trimmed and i == len(decisions) - 1:
            action_entry["trim_strategy"] = decision.get("extra_args", "")
            action_entry["final_gap_after_trim"] = round(final_gap, 4)

        actions_tried.append(action_entry)

    lesson = ""
    if esm_post_applied:
        delta = esm_gap_before - esm_gap_after
        lesson = (
            f"ESM post-correction gap {esm_gap_before:.2f} -> {esm_gap_after:.2f} "
            f"(delta {delta:+.3f})"
        )
    elif was_trimmed:
        lesson = f"Used trimal to reduce gap_fraction from {initial_gap:.2f} to {final_gap:.2f}"
    elif final_gap < 0.1:
        lesson = f"MAFFT params improved alignment, final gap_fraction: {final_gap:.2f}"
    else:
        lesson = "Could not improve alignment significantly"

    sanitized_metrics = _sanitize_metrics(final_metrics)

    record = {
        "experiment_id": experiment_id,
        "timestamp": datetime.utcnow().isoformat(),
        "num_sequences": num_sequences,
        "avg_sequence_length": round(avg_seq_len, 1),
        "sequence_signature": list(sequence_signature),
        "tfidf_embedding": tfidf_embedding,
        "fingerprint": fingerprint,
        "problem_type": problem_type,
        "initial_gap_fraction": round(initial_gap, 4),
        "final_gap_fraction": round(final_gap, 4),
        "extended_metrics": sanitized_metrics,
        "actions_tried": actions_tried,
        "was_trimmed": was_trimmed,
        "final_outcome": _classify_outcome(initial_gap, final_gap, was_trimmed),
        "lesson": lesson,
        "esm_post_applied": esm_post_applied,
        "esm_gap_before": round(esm_gap_before, 4) if esm_post_applied else None,
        "esm_gap_after": round(esm_gap_after, 4) if esm_post_applied else None,
    }

    _append_record(record)

    existing_matrix, existing_vocab, existing_ids = load_index()
    new_matrix, new_vocab, new_ids = add_to_index(
        existing_matrix, existing_vocab, existing_ids,
        experiment_id, tfidf_embedding
    )
    save_index(new_matrix, new_vocab, new_ids)

    update_strategy_stats(record)

    from rag_retrieve import update_esm_gap_stats, update_esm_post_stats
    update_esm_gap_stats(record)
    update_esm_post_stats(record)

    print(f"  -> Stored to knowledge base: {JSONL_GZ_FILE} + {INDEX_SUFFIX}")