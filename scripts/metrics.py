import re
from typing import Optional


def compute_metrics(alignment_text: str) -> dict:
    if not alignment_text or not alignment_text.strip():
        return {
            "gap_fraction": 0.0,
            "alignment_length": 0,
            "avg_sequence_length": 0.0
        }

    sequences = []
    lines = alignment_text.strip().splitlines()
    current_header = None
    current_seq = []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if current_header is not None:
                sequences.append("".join(current_seq))
            current_header = line[1:]
            current_seq = []
        else:
            if current_header is None and not sequences:
                continue
            current_seq.append(line)

    if current_header is not None or current_seq:
        sequences.append("".join(current_seq))

    if not sequences:
        return {
            "gap_fraction": 0.0,
            "alignment_length": 0,
            "avg_sequence_length": 0.0
        }

    alignment_length = len(sequences[0])
    for seq in sequences[1:]:
        if len(seq) != alignment_length:
            alignment_length = max(alignment_length, len(seq))

    total_chars = 0
    total_gaps = 0

    for seq in sequences:
        seq_len = len(seq)
        total_chars += seq_len
        total_gaps += seq.count("-")

    gap_fraction = total_gaps / total_chars if total_chars > 0 else 0.0
    avg_sequence_length = total_chars / len(sequences) if sequences else 0.0

    return {
        "gap_fraction": float(gap_fraction),
        "alignment_length": int(alignment_length),
        "avg_sequence_length": float(avg_sequence_length)
    }