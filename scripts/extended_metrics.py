import math
import re
from collections import Counter
from typing import Dict, List, Optional


def parse_sequences(alignment: str) -> List[str]:
    sequences = []
    current_seq = []

    for line in alignment.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if current_seq:
                sequences.append("".join(current_seq))
            current_seq = []
        else:
            current_seq.append(line)

    if current_seq:
        sequences.append("".join(current_seq))

    return sequences


def _compute_hill_number(counts: List[int], q: float) -> float:
    total = sum(counts)
    if total == 0:
        return 0.0
    if q == 1.0:
        entropy = 0.0
        for c in counts:
            if c > 0:
                p = c / total
                entropy -= p * math.log(p)
        return math.exp(entropy)
    power_sum = sum((c / total) ** q for c in counts if c > 0)
    if power_sum <= 0:
        return 0.0
    return power_sum ** (1.0 / (1.0 - q))


def compute_hill_numbers_per_column(alignment: str) -> List[Dict]:
    sequences = parse_sequences(alignment)
    if not sequences:
        return []

    aln_len = len(sequences[0])
    results = []

    for col_idx in range(aln_len):
        column = [seq[col_idx] for seq in sequences if col_idx < len(seq)]
        total_in_col = len(column)
        gaps = column.count("-")
        residues = [c for c in column if c != "-"]

        if len(residues) < 2:
            results.append({
                "column": col_idx + 1,
                "hill_q0": 0.0,
                "hill_q1": 0.0,
                "hill_q2": 0.0,
                "gap_fraction": round(gaps / total_in_col, 3) if total_in_col > 0 else 1.0,
                "num_residues": len(residues),
            })
            continue

        counter = Counter(residues)
        counts = list(counter.values())
        dominant = counter.most_common(1)[0]

        results.append({
            "column": col_idx + 1,
            "hill_q0": round(_compute_hill_number(counts, 0.0), 3),
            "hill_q1": round(_compute_hill_number(counts, 1.0), 3),
            "hill_q2": round(_compute_hill_number(counts, 2.0), 3),
            "gap_fraction": round(gaps / total_in_col, 3) if total_in_col > 0 else 1.0,
            "num_residues": len(residues),
            "dominant_residue": dominant[0],
            "dominant_fraction": round(dominant[1] / len(residues), 3),
        })

    return results


def compute_hill_metrics(alignment: str) -> Dict:
    column_data = compute_hill_numbers_per_column(alignment)
    if not column_data:
        return {
            "hill_q0_avg": 0.0,
            "hill_q1_avg": 0.0,
            "hill_q2_avg": 0.0,
            "hill_q0_max": 0.0,
            "hill_q1_max": 0.0,
            "problematic_columns": [],
        }

    q0_vals = [c["hill_q0"] for c in column_data]
    q1_vals = [c["hill_q1"] for c in column_data]
    q2_vals = [c["hill_q2"] for c in column_data]

    problematic = []
    for c in column_data:
        if c.get("num_residues", 0) < 2:
            continue
        if c["hill_q1"] > 5.0 or c["gap_fraction"] > 0.3:
            problematic.append({
                "column": c["column"],
                "hill_q1": c["hill_q1"],
                "gap_fraction": c["gap_fraction"],
                "num_residues": c["num_residues"],
            })

    return {
        "hill_q0_avg": round(sum(q0_vals) / len(q0_vals), 3),
        "hill_q1_avg": round(sum(q1_vals) / len(q1_vals), 3),
        "hill_q2_avg": round(sum(q2_vals) / len(q2_vals), 3),
        "hill_q0_max": round(max(q0_vals), 3),
        "hill_q1_max": round(max(q1_vals), 3),
        "problematic_columns": problematic[:10],
    }


def compute_sp_score(alignment: str) -> float:
    sequences = parse_sequences(alignment)
    if not sequences or len(sequences) < 2:
        return 0.0

    aln_len = len(sequences[0])
    if aln_len == 0:
        return 0.0

    total_score = 0
    valid_columns = 0

    for col_idx in range(aln_len):
        column = [seq[col_idx] for seq in sequences if col_idx < len(seq)]
        column = [c for c in column if c != "-"]

        if len(column) < 2:
            continue

        valid_columns += 1
        matches = sum(1 for i in range(len(column)) for j in range(i+1, len(column)) if column[i] == column[j])
        total_pairs = (len(column) * (len(column) - 1)) / 2

        if total_pairs > 0:
            total_score += matches / total_pairs

    return total_score / valid_columns if valid_columns > 0 else 0.0


def compute_cs_score(alignment: str) -> float:
    sequences = parse_sequences(alignment)
    if not sequences:
        return 0.0

    aln_len = len(sequences[0])
    if aln_len == 0:
        return 0.0

    conserved_columns = 0
    valid_columns = 0

    for col_idx in range(aln_len):
        column = [seq[col_idx] for seq in sequences if col_idx < len(seq)]
        column = [c for c in column if c != "-"]

        if len(column) < 2:
            continue

        valid_columns += 1
        if len(set(column)) == 1:
            conserved_columns += 1

    return conserved_columns / valid_columns if valid_columns > 0 else 0.0


def compute_conservation(alignment: str) -> float:
    sequences = parse_sequences(alignment)
    if not sequences:
        return 0.0

    aln_len = len(sequences[0])
    if aln_len == 0:
        return 0.0

    entropy_sum = 0
    valid_columns = 0

    for col_idx in range(aln_len):
        column = [seq[col_idx] for seq in sequences if col_idx < len(seq)]
        column = [c for c in column if c != "-"]

        if not column:
            continue

        valid_columns += 1
        total = len(column)
        counts = Counter(column)
        entropy = 0

        for count in counts.values():
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)

        max_entropy = math.log2(min(len(counts), 20))
        if max_entropy > 0:
            entropy_sum += 1 - (entropy / max_entropy)

    return entropy_sum / valid_columns if valid_columns > 0 else 0.0


def compute_gap_distribution(alignment: str) -> float:
    sequences = parse_sequences(alignment)
    if not sequences:
        return 0.0

    aln_len = len(sequences[0])
    if aln_len == 0:
        return 0.0

    gap_counts_per_col = []
    for col_idx in range(aln_len):
        gaps = sum(1 for seq in sequences if col_idx < len(seq) and seq[col_idx] == "-")
        gap_counts_per_col.append(gaps / len(sequences))

    if not gap_counts_per_col:
        return 0.0

    mean = sum(gap_counts_per_col) / len(gap_counts_per_col)
    variance = sum((x - mean) ** 2 for x in gap_counts_per_col) / len(gap_counts_per_col)

    return math.sqrt(variance)


def compute_identity(alignment: str) -> float:
    sequences = parse_sequences(alignment)
    if not sequences or len(sequences) < 2:
        return 0.0

    aln_len = len(sequences[0])

    total_identity = 0
    comparisons = 0

    for i in range(len(sequences)):
        for j in range(i + 1, len(sequences)):
            matches = 0
            valid = 0
            for col in range(aln_len):
                a = sequences[i][col] if col < len(sequences[i]) else "-"
                b = sequences[j][col] if col < len(sequences[j]) else "-"
                if a != "-" and b != "-":
                    valid += 1
                    if a == b:
                        matches += 1
            if valid > 0:
                total_identity += matches / valid
                comparisons += 1

    return total_identity / comparisons if comparisons > 0 else 0.0


def compute_extended_metrics(alignment: str) -> Dict:
    sequences = parse_sequences(alignment)

    gap_count = sum(row.count("-") for row in sequences)
    total_chars = sum(len(row) for row in sequences)
    gap_fraction = gap_count / total_chars if total_chars > 0 else 0.0

    alignment_length = len(sequences[0]) if sequences else 0
    avg_seq_length = total_chars / len(sequences) if sequences else 0

    hill = compute_hill_metrics(alignment)

    return {
        "gap_fraction": round(gap_fraction, 4),
        "alignment_length": alignment_length,
        "avg_sequence_length": round(avg_seq_length, 2),
        "sp_score": round(compute_sp_score(alignment), 4),
        "cs_score": round(compute_cs_score(alignment), 4),
        "conservation": round(compute_conservation(alignment), 4),
        "gap_distribution": round(compute_gap_distribution(alignment), 4),
        "identity": round(compute_identity(alignment), 4),
        "hill_q0_avg": hill["hill_q0_avg"],
        "hill_q1_avg": hill["hill_q1_avg"],
        "hill_q2_avg": hill["hill_q2_avg"],
        "hill_q0_max": hill["hill_q0_max"],
        "hill_q1_max": hill["hill_q1_max"],
        "hill_problematic_columns": hill["problematic_columns"],
    }