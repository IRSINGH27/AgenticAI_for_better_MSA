import math
import pathlib
import tempfile
import os
from typing import Dict, List, Optional, Tuple

import torch

from protein_embeddings import (
    compute_residue_embeddings,
    cosine_similarity_matrix,
    is_protein_sequence,
)

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()

ESM_STRATEGIES = {
    "local": "Local alignment using ESM-2 embedding similarity (default)",
    "global": "Global alignment using ESM-2 embedding similarity",
}

ESM_FLAGS = {
    "gap_open": "Gap opening penalty (default: -2.0)",
    "gap_extend": "Gap extension penalty (default: -0.5)",
}


def _needleman_wunsch(seq1: str, seq2: str, embs1: torch.Tensor, embs2: torch.Tensor,
                      gap_open: float = -2.0, gap_extend: float = -0.5) -> Tuple[str, str, float]:
    n = len(seq1)
    m = len(seq2)
    gap_penalty = gap_open

    res_sim = cosine_similarity_matrix(embs1, embs2) if embs1.numel() > 0 and embs2.numel() > 0 else torch.zeros(n, m)

    score = [[0.0] * (m + 1) for _ in range(n + 1)]
    trace = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        score[i][0] = i * gap_penalty
        trace[i][0] = 1
    for j in range(1, m + 1):
        score[0][j] = j * gap_penalty
        trace[0][j] = 2

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            match = score[i-1][j-1] + res_sim[i-1, j-1].item()
            delete = score[i-1][j] + gap_penalty
            insert = score[i][j-1] + gap_penalty

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


def _star_align(sequences: List[str], residue_embs: List[torch.Tensor],
                gap_open: float = -2.0, gap_extend: float = -0.5) -> List[str]:
    n = len(sequences)
    if n == 0:
        return []
    if n == 1:
        return [sequences[0]]

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
    # Center as column lists (chars + embeddings); grows as sequences are added
    center_cols: List[str] = list(sequences[center])
    center_emb_cols: List[torch.Tensor] = [center_embs[k]
                                           for k in range(center_embs.shape[0])]
    # Each other sequence aligned against the center, tracked by column index
    others: Dict[int, List[str]] = {}

    for i in range(n):
        if i == center:
            continue

        # NW between current gapped center and the new sequence
        c_str = "".join(center_cols)
        c_emb = (torch.stack(center_emb_cols) if center_emb_cols
                 else torch.zeros(0, residue_embs[i].shape[1]
                                  if residue_embs[i].numel() else 1))
        a1, a2, _ = _needleman_wunsch(
            c_str, sequences[i], c_emb, residue_embs[i],
            gap_open, gap_extend,
        )
        # a1 contains every old center column in order, plus insertion '-'s.
        # Merge: old column q -> same column for everyone; pure '-' -> new
        # insertion column (everyone else gets '-').
        new_center: List[str] = []
        new_center_embs: List[torch.Tensor] = []
        remap: List[Optional[int]] = []   # a1 position -> old center col (or None)
        q = 0
        zero_emb = torch.zeros(c_emb.shape[1] if c_emb.numel() else 0)
        for k, ch in enumerate(a1):
            if q < len(center_cols) and ch == center_cols[q]:
                new_center.append(center_cols[q])
                new_center_embs.append(center_emb_cols[q])
                remap.append(q)
                q += 1
            elif ch == "-":
                new_center.append("-")
                new_center_embs.append(zero_emb)
                remap.append(None)
            else:
                # traceback must preserve seq1 order; be defensive anyway
                new_center.append(ch)
                new_center_embs.append(zero_emb)
                remap.append(None)
        # Any leftover old columns (shouldn't happen) keep their place
        while q < len(center_cols):
            new_center.append(center_cols[q])
            new_center_embs.append(center_emb_cols[q])
            remap.append(q)
            q += 1

        # Shift every existing sequence through the same remap
        for old_idx, cols in others.items():
            shifted: List[str] = []
            for r in remap:
                shifted.append(cols[r] if r is not None and r < len(cols) else "-")
            others[old_idx] = shifted
        others[i] = list(a2)

        center_cols = new_center
        center_emb_cols = new_center_embs

    # Assemble rows in ORIGINAL sequence order (run_esm_alignment indexes by id)
    rows_by_idx: Dict[int, List[str]] = {center: center_cols}
    rows_by_idx.update(others)
    length = len(center_cols)
    return ["".join(rows_by_idx.get(i, [])[:length]).ljust(length, "-")
            for i in range(n)]


def run_esm_alignment(fasta_text: str, params: Optional[dict] = None) -> str:
    if params is None:
        params = {}

    strategy = params.get("strategy", "local")
    gap_open = float(params.get("gap_open", -2.0))
    gap_extend = float(params.get("gap_extend", -0.5))

    entries = []
    current_id = None
    current_seq = []
    for line in fasta_text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if current_id is not None:
                entries.append((current_id, "".join(current_seq)))
            current_id = line[1:].split()[0]
            current_seq = []
        else:
            current_seq.append(line.upper())
    if current_id is not None:
        entries.append((current_id, "".join(current_seq)))

    if not entries:
        return ""

    if len(entries) == 1:
        return f">{entries[0][0]}\n{entries[0][1]}"

    seq_ids = [e[0] for e in entries]
    sequences = [e[1] for e in entries]

    print(f"  ESM-2: Computing residue embeddings for {len(sequences)} sequences...")
    residue_data = compute_residue_embeddings(fasta_text)

    residue_embs = [entry["embeddings"] for entry in residue_data]

    print(f"  ESM-2: Running star alignment (strategy={strategy})...")
    aligned_seqs = _star_align(sequences, residue_embs, gap_open, gap_extend)

    output_lines = []
    for idx, seq_id in enumerate(seq_ids):
        output_lines.append(f">{seq_id}")
        aligned = aligned_seqs[idx] if idx < len(aligned_seqs) else ""
        for i in range(0, len(aligned), 80):
            output_lines.append(aligned[i:i+80])

    return "\n".join(output_lines)


ESM_ALIGNER_INFO = {
    "tool": "esm2",
    "description": "ESM-2 embedding-based alignment (deep learning, good for divergent proteins)",
    "strategies": ESM_STRATEGIES,
    "flags": ESM_FLAGS,
}
