from rag_retrieve import retrieve_relevant, format_retrieved
from typing import Optional


def build_rag_context(
    gap_fraction: float,
    problem_type: Optional[str] = None,
    exclude_experiment_id: Optional[str] = None,
    fasta_text: Optional[str] = None,
    top_k: int = 3
) -> str:
    if problem_type is None:
        if gap_fraction > 0.2:
            problem_type = "high_gap_fraction"
        elif gap_fraction > 0.1:
            problem_type = "moderate_gap_fraction"
        else:
            problem_type = "good_quality"

    relevant = retrieve_relevant(
        problem_type=problem_type,
        gap_fraction=gap_fraction,
        exclude_id=exclude_experiment_id,
        top_k=top_k,
        fasta_text=fasta_text
    )

    return format_retrieved(relevant, problem_type)