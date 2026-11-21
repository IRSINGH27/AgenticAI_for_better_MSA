import math
from typing import Dict, List, Optional, Tuple

import torch

from protein_embeddings import compute_residue_probabilities
from extended_metrics import parse_sequences, compute_hill_numbers_per_column


def _compute_column_esm_conservation(alignment: str, prob_data: List[Dict]) -> List[Dict]:
    sequences = parse_sequences(alignment)
    if not sequences:
        return []

    aln_len = len(sequences[0])
    id_to_probs = {entry["id"]: entry for entry in prob_data}

    columns = []
    for col_idx in range(aln_len):
        col_residues = []
        col_conservations = []
        col_entropies = []

        for seq_idx, seq in enumerate(sequences):
            if col_idx >= len(seq):
                continue
            residue = seq[col_idx]
            if residue == "-":
                continue

            seq_id = None
            for entry in prob_data:
                if entry["sequence"].replace("-", "") == seq.replace("-", ""):
                    seq_id = entry["id"]
                    break

            if seq_id and seq_id in id_to_probs:
                probs = id_to_probs[seq_id]
                orig_pos = col_idx
                if col_idx < len(probs["conservation_score"]):
                    col_residues.append(residue)
                    col_conservations.append(probs["conservation_score"][orig_pos])
                    col_entropies.append(probs["entropy"][orig_pos])

        if col_conservations:
            columns.append({
                "column": col_idx + 1,
                "esm_conservation": round(sum(col_conservations) / len(col_conservations), 4),
                "esm_entropy": round(sum(col_entropies) / len(col_entropies), 4),
                "num_residues": len(col_residues),
            })
        else:
            columns.append({
                "column": col_idx + 1,
                "esm_conservation": 0.0,
                "esm_entropy": 0.0,
                "num_residues": 0,
            })

    return columns


def _pearson_correlation(xs: List[float], ys: List[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0

    mean_x = sum(xs) / n
    mean_y = sum(ys) / n

    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    std_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    std_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))

    if std_x == 0 or std_y == 0:
        return 0.0

    return cov / (std_x * std_y)


def analyze_conservation(alignment: str, prob_data: List[Dict],
                         conservation_threshold: float = 0.7,
                         hill_threshold: float = 5.0) -> Dict:
    esm_cols = _compute_column_esm_conservation(alignment, prob_data)
    hill_cols = compute_hill_numbers_per_column(alignment)

    paired = []
    for esm_col, hill_col in zip(esm_cols, hill_cols):
        if esm_col["num_residues"] >= 2 and hill_col.get("num_residues", 0) >= 2:
            paired.append((esm_col, hill_col))

    if not paired:
        return {
            "conservation_correlation": 0.0,
            "columns_analyzed": 0,
            "esm_misaligned_columns": [],
            "conserved_columns": 0,
            "variable_columns": 0,
        }

    esm_vals = [p[0]["esm_conservation"] for p in paired]
    observed_vals = [1.0 - min(p[1].get("hill_q1", 0) / 10.0, 1.0) for p in paired]

    correlation = _pearson_correlation(esm_vals, observed_vals)

    misaligned = []
    conserved_count = 0
    variable_count = 0

    for esm_col, hill_col in paired:
        esm_score = esm_col["esm_conservation"]
        hill_q1 = hill_col.get("hill_q1", 0)

        if esm_score > conservation_threshold:
            conserved_count += 1
        elif esm_score < 0.3:
            variable_count += 1

        if esm_score > conservation_threshold and hill_q1 > hill_threshold:
            misaligned.append({
                "column": esm_col["column"],
                "esm_conservation": esm_score,
                "observed_hill_q1": hill_q1,
                "gap_fraction": hill_col.get("gap_fraction", 0),
            })

    return {
        "conservation_correlation": round(correlation, 4),
        "columns_analyzed": len(paired),
        "esm_misaligned_columns": misaligned[:10],
        "conserved_columns": conserved_count,
        "variable_columns": variable_count,
    }


def _compute_position_gap_penalties(sequences: List[str], prob_data: List[Dict],
                                     gap_open: float, alpha: float) -> torch.Tensor:
    max_len = max(len(s) for s in sequences) if sequences else 0
    penalties = torch.full((max_len,), gap_open)

    id_to_probs = {entry["id"]: entry for entry in prob_data}

    for seq in sequences:
        for pos in range(len(seq)):
            if pos >= max_len:
                break
            for entry in prob_data:
                if entry["sequence"].replace("-", "") == seq.replace("-", ""):
                    if pos < len(entry["conservation_score"]):
                        cons = entry["conservation_score"][pos]
                        adjusted = gap_open * (1.0 + alpha * cons)
                        penalties[pos] = min(penalties[pos].item(), adjusted)
                    break

    return penalties


def _needleman_wunsch_with_position_gaps(
    seq1: str, seq2: str,
    embs1: torch.Tensor, embs2: torch.Tensor,
    gap_penalties1: torch.Tensor, gap_penalties2: torch.Tensor,
    gap_extend: float = -0.5
) -> Tuple[str, str, float]:
    from protein_embeddings import cosine_similarity_matrix

    n = len(seq1)
    m = len(seq2)

    res_sim = cosine_similarity_matrix(embs1, embs2) if embs1.numel() > 0 and embs2.numel() > 0 else torch.zeros(n, m)

    score = [[0.0] * (m + 1) for _ in range(n + 1)]
    trace = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        score[i][0] = score[i-1][0] + gap_penalties1[i-1].item()
        trace[i][0] = 1
    for j in range(1, m + 1):
        score[0][j] = score[0][j-1] + gap_penalties2[j-1].item()
        trace[0][j] = 2

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            match = score[i-1][j-1] + res_sim[i-1, j-1].item()
            delete = score[i-1][j] + gap_penalties1[i-1].item()
            insert = score[i][j-1] + gap_penalties2[j-1].item()

            if match >= delete and match >= insert:
                score[i][j] = match
                trace[i][j] = 0
            elif delete >= insert:
                score[i][j] = delete
                trace[i][j] = 1
            else:
                score[i][j] = insert
                trace[i][j] = 2

    aligned1 = []
    aligned2 = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and trace[i][j] == 0:
            aligned1.append(seq1[i-1])
            aligned2.append(seq2[j-1])
            i -= 1
            j -= 1
        elif i > 0 and trace[i][j] == 1:
            aligned1.append(seq1[i-1])
            aligned2.append("-")
            i -= 1
        else:
            aligned1.append("-")
            aligned2.append(seq2[j-1])
            j -= 1

    aligned1.reverse()
    aligned2.reverse()
    return "".join(aligned1), "".join(aligned2), score[n][m]


def refine_gaps(fasta_text: str, alignment: str, alpha: float = 2.0,
                gap_open: float = -2.0, gap_extend: float = -0.5) -> Tuple[str, Dict]:
    from protein_embeddings import compute_residue_embeddings

    sequences = parse_sequences(alignment)
    if not sequences or len(sequences) < 2:
        return alignment, {"gap_refined": False, "gap_changes": 0}

    print(f"  ESM refine: Computing residue embeddings...")
    residue_data = compute_residue_embeddings(fasta_text)
    residue_embs = [entry["embeddings"] for entry in residue_data]

    print(f"  ESM refine: Computing residue probabilities...")
    prob_data = compute_residue_probabilities(fasta_text)

    print(f"  ESM refine: Computing position-specific gap penalties (alpha={alpha})...")
    clean_seqs = [s.replace("-", "") for s in sequences]
    gap_penalties = _compute_position_gap_penalties(clean_seqs, prob_data, gap_open, alpha)

    original_gaps = sum(s.count("-") for s in sequences)

    from protein_embeddings import cosine_similarity_matrix

    n = len(clean_seqs)
    if n == 0:
        return alignment, {"gap_refined": False, "gap_changes": 0}

    seq_sims = torch.zeros(n, n)
    for i in range(n):
        for j in range(i + 1, n):
            if residue_embs[i].numel() > 0 and residue_embs[j].numel() > 0:
                sim = cosine_similarity_matrix(residue_embs[i], residue_embs[j])
                seq_sims[i, j] = sim.mean().item()
                seq_sims[j, i] = seq_sims[i, j]
        seq_sims[i, i] = 1.0

    total_scores = []
    for i in range(n):
        total = sum(seq_sims[i, j].item() for j in range(n) if i != j)
        total_scores.append(total)

    center = total_scores.index(max(total_scores))
    center_embs = residue_embs[center]
    aligned = [list(clean_seqs[center])]

    for i in range(n):
        if i == center:
            continue

        center_penalties = gap_penalties[:len(aligned[0])]
        seq_penalties = gap_penalties[:len(clean_seqs[i])]

        a1, a2, _ = _needleman_wunsch_with_position_gaps(
            "".join(aligned[0]), clean_seqs[i],
            center_embs, residue_embs[i],
            center_penalties, seq_penalties,
            gap_extend
        )

        new_aligned = [list(a1)]

        new_center_embs = torch.zeros(len(a1), center_embs.shape[1] if center_embs.numel() > 0 else 1)
        orig_idx = 0
        for k, c in enumerate(a1):
            if c == "-":
                new_center_embs[k] = 0
            else:
                if orig_idx < center_embs.shape[0]:
                    new_center_embs[k] = center_embs[orig_idx]
                orig_idx += 1
        center_embs = new_center_embs

        for existing in aligned[1:]:
            new_existing = []
            orig_idx = 0
            for k, c in enumerate(a1):
                if c == "-":
                    new_existing.append(existing[orig_idx] if orig_idx < len(existing) else "-")
                else:
                    new_existing.append(existing[orig_idx] if orig_idx < len(existing) else "-")
                    orig_idx += 1
            while orig_idx < len(existing):
                new_existing.append(existing[orig_idx])
                orig_idx += 1
            new_aligned.append(new_existing)

        new_aligned.append(list(a2))
        aligned = new_aligned

    refined_seqs = ["".join(col) for col in zip(*aligned)]
    new_gaps = sum(s.count("-") for s in refined_seqs)
    gap_changes = abs(new_gaps - original_gaps)

    output_lines = []
    for idx, seq in enumerate(refined_seqs):
        output_lines.append(f">seq_{idx}")
        for i in range(0, len(seq), 80):
            output_lines.append(seq[i:i+80])

    refined_alignment = "\n".join(output_lines)

    return refined_alignment, {
        "gap_refined": True,
        "original_gaps": original_gaps,
        "new_gaps": new_gaps,
        "gap_changes": gap_changes,
        "alpha": alpha,
    }


def generate_column_report(analysis: Dict, alignment: str) -> str:
    lines = []
    lines.append("=== ESM Conservation Analysis ===")
    lines.append("")
    corr = analysis.get("conservation_correlation", 0)
    if corr > 0.7:
        agreement = "ESM agrees with alignment"
    elif corr > 0.3:
        agreement = "ESM partially agrees with alignment"
    else:
        agreement = "ESM disagrees with alignment"
    lines.append(f"Overall correlation: r = {corr:.2f} ({agreement})")
    lines.append("")
    lines.append(f"Columns analyzed: {analysis.get('columns_analyzed', 0)}")
    lines.append(f"Conserved columns (ESM score > 0.7): {analysis.get('conserved_columns', 0)}")
    lines.append(f"Variable columns (ESM score < 0.3): {analysis.get('variable_columns', 0)}")

    misaligned = analysis.get("esm_misaligned_columns", [])
    lines.append(f"Misaligned columns (ESM disagrees): {len(misaligned)}")

    if misaligned:
        lines.append("")
        lines.append("Misaligned columns (ESM says conserved, alignment shows diversity):")
        for col in misaligned[:10]:
            lines.append(
                f"  Col {col['column']}: ESM={col['esm_conservation']:.2f}, "
                f"Hill_q1={col['observed_hill_q1']:.1f}, "
                f"gaps={col['gap_fraction']:.0%}"
            )
        lines.append("")
        lines.append("Recommendation: Consider re-alignment or trimming these columns")

    return "\n".join(lines)


def run_esm_analysis(fasta_text: str, alignment: str,
                     refine_gaps_flag: bool = False,
                     alpha: float = 2.0) -> Dict:
    print("\n  Running ESM probability analysis...")
    prob_data = compute_residue_probabilities(fasta_text)

    print("  Comparing ESM conservation vs observed conservation...")
    analysis = analyze_conservation(alignment, prob_data)

    final_alignment = alignment
    gap_info = {"gap_refined": False, "gap_changes": 0}

    if refine_gaps_flag:
        print("  Refining gap placement with ESM conservation scores...")
        final_alignment, gap_info = refine_gaps(fasta_text, alignment, alpha=alpha)
        analysis.update(gap_info)

    report = generate_column_report(analysis, final_alignment)
    print(f"\n{report}\n")

    return {
        "analysis": analysis,
        "report": report,
        "final_alignment": final_alignment,
    }
