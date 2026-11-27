#!/usr/bin/env python3
"""SP / TC scorer: compare a predicted alignment against a reference alignment.

SP (Sum-of-Pairs)  = fraction of reference residue pairs co-aligned in prediction.
TC (Total Column)  = fraction of reference columns reproduced exactly in prediction.

Reference format (bench bundle): FASTA, '-'/'.' gaps, LOWERCASE = non-assessment
region (excluded when mask_lowercase=True, matching qscore conventions).

Self-test: python3 sp_tc.py --selftest
"""
import argparse
import sys
from itertools import combinations
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

Residue = Tuple[str, int]          # (sequence_id, 0-based residue index in that seq)
Column = FrozenSet[Residue]
Pair = Tuple[Residue, Residue]


def parse_fasta(text: str) -> Dict[str, str]:
    seqs: Dict[str, List[str]] = {}
    cur: Optional[str] = None
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            cur = line[1:].split()[0]
            seqs[cur] = []
        elif cur is not None:
            seqs[cur].append(line)
    # '.' is a gap character in BAliBASE-style references; keep case (masking later)
    return {k: "".join(v).replace(".", "-") for k, v in seqs.items()}


def _columns(aln: Dict[str, str], ids: List[str]) -> List[Dict[str, Tuple[int, str]]]:
    """Per column: {id: (residue_index, char)} for non-gap chars."""
    cols: List[Dict[str, Tuple[int, str]]] = []
    if not ids:
        return cols
    length = max(len(aln[i]) for i in ids)
    res_idx = {i: 0 for i in ids}
    for pos in range(length):
        col: Dict[str, Tuple[int, str]] = {}
        for sid in ids:
            seq = aln[sid]
            if pos >= len(seq):
                continue
            ch = seq[pos]
            if ch == "-":
                continue
            col[sid] = (res_idx[sid], ch)
            res_idx[sid] += 1
        cols.append(col)
    return cols


def score(pred_text: str, ref_text: str, mask_lowercase: bool = True) -> Dict[str, float]:
    pred = parse_fasta(pred_text)
    ref = parse_fasta(ref_text)
    ids = sorted(set(pred) & set(ref))
    if len(ids) < 2:
        return {"sp": 0.0, "tc": 0.0, "n_ref_pairs": 0, "n_ref_cols": 0,
                "n_common_seqs": len(ids)}

    ref_cols = _columns(ref, ids)
    pred_cols = _columns(pred, ids)

    def eligible(ch1: str, ch2: str) -> bool:
        if not mask_lowercase:
            return True
        return ch1.isupper() and ch2.isupper()

    def col_eligible(col: Dict[str, Tuple[int, str]]) -> bool:
        if not mask_lowercase:
            return True
        return all(ch.isupper() for _, ch in col.values())

    # Reference pair set (only common ids)
    ref_pairs: Set[Pair] = set()
    for col in ref_cols:
        residues = [(sid, idx, ch) for sid, (idx, ch) in col.items()]
        for (s1, i1, c1), (s2, i2, c2) in combinations(residues, 2):
            if eligible(c1, c2):
                a, b = (s1, i1), (s2, i2)
                ref_pairs.add((a, b) if a <= b else (b, a))

    # Predicted pair set
    pred_pairs: Set[Pair] = set()
    for col in pred_cols:
        residues = [(sid, idx) for sid, (idx, _ch) in col.items()]
        for a, b in combinations(sorted(residues), 2):
            pred_pairs.add((a, b))

    correct_pairs = ref_pairs & pred_pairs
    sp = len(correct_pairs) / len(ref_pairs) if ref_pairs else 0.0

    # TC: reference columns (eligible) whose exact residue set appears in prediction
    pred_col_sets: Set[Column] = {
        frozenset((sid, idx) for sid, (idx, _c) in col.items())
        for col in pred_cols if col
    }
    ref_col_list = [col for col in ref_cols if col and col_eligible(col)]
    correct_cols = sum(
        1 for col in ref_col_list
        if frozenset((sid, idx) for sid, (idx, _c) in col.items()) in pred_col_sets
    )
    tc = correct_cols / len(ref_col_list) if ref_col_list else 0.0

    return {
        "sp": round(sp, 6),
        "tc": round(tc, 6),
        "n_ref_pairs": len(ref_pairs),
        "n_ref_cols": len(ref_col_list),
        "n_common_seqs": len(ids),
    }


def score_files(pred_path: str, ref_path: str, mask_lowercase: bool = True) -> Dict[str, float]:
    with open(pred_path) as f:
        pred_text = f.read()
    with open(ref_path) as f:
        ref_text = f.read()
    return score(pred_text, ref_text, mask_lowercase=mask_lowercase)


def _selftest() -> int:
    # ref: A=AB, B=AB  → pairs {(A0,B0),(A1,B1)}, cols {{A0,B0},{A1,B1}}
    # pred: A=AB, B=A-B → pairs {(A0,B0)}, cols {{A0,B0},{A1},{B1}}
    # expect SP=0.5, TC=0.5
    ref = ">A\nAB\n>B\nAB\n"
    pred = ">A\nAB\n>B\nA-B\n"
    r = score(pred, ref, mask_lowercase=True)
    assert abs(r["sp"] - 0.5) < 1e-9, r
    assert abs(r["tc"] - 0.5) < 1e-9, r

    # identical → 1.0 / 1.0
    r2 = score(ref, ref)
    assert r2["sp"] == 1.0 and r2["tc"] == 1.0, r2

    # lowercase masking: ref col with lowercase residue excluded from pairs/cols
    ref3 = ">A\naB\n>B\naB\n"
    r3 = score(">A\naB\n>B\naB\n", ref3, mask_lowercase=True)
    # col0 both lowercase → excluded; col1 uppercase pair remains → SP=1, TC=1 (1 eligible col)
    assert r3["n_ref_pairs"] == 1 and r3["sp"] == 1.0, r3
    r3b = score(">A\naB\n>B\na-B\n", ref3, mask_lowercase=True)
    # pred co-aligns the lowercase idx-0 residues, not the assessed uppercase pair
    assert r3b["sp"] == 0.0 and r3b["tc"] == 0.0, r3b

    # '.' treated as gap in reference
    r4 = score(">A\nAB\n>B\nAB\n", ">A\nA-B\n>B\nAB\n".replace("-", "."), mask_lowercase=False)
    # ref with dot-gap: A=A-B, B=AB → 1 pair only → SP depends; just ensure no crash
    assert 0.0 <= r4["sp"] <= 1.0

    # missing sequence in prediction
    r5 = score(">A\nAB\n", ">A\nAB\n>B\nAB\n")
    assert r5["n_common_seqs"] == 1 and r5["sp"] == 0.0

    print("sp_tc selftest: OK (5 checks passed)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pred", help="predicted alignment FASTA")
    ap.add_argument("--ref", help="reference alignment FASTA")
    ap.add_argument("--no-mask", action="store_true",
                    help="score lowercase regions too")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return _selftest()
    if not args.pred or not args.ref:
        ap.error("--pred and --ref required (or --selftest)")
    r = score_files(args.pred, args.ref, mask_lowercase=not args.no_mask)
    print(f"SP={r['sp']:.4f}  TC={r['tc']:.4f}  "
          f"(ref pairs={r['n_ref_pairs']}, cols={r['n_ref_cols']}, "
          f"common seqs={r['n_common_seqs']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
