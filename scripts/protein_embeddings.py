import math
import pathlib
import re
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import EsmModel, EsmForMaskedLM, EsmTokenizer

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
MODEL_CACHE = SCRIPT_DIR.parent / "models" / "models--facebook--esm2_t33_650M_UR50D"
MODEL_NAME = "facebook/esm2_t33_650M_UR50D"

AA_TOKENS = list("ACDEFGHIKLMNPQRSTVWY")


def _resolve_model_path():
    snapshot_dir = MODEL_CACHE / "snapshots"
    if snapshot_dir.exists():
        snapshots = list(snapshot_dir.iterdir())
        if snapshots:
            return snapshots[0]
    return None


MODEL_DIR = _resolve_model_path()

_model = None
_tokenizer = None
_mlm_model = None


def _load_model():
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer

    if MODEL_DIR.exists():
        print("  Loading ESM-2 model from local cache...")
        _tokenizer = EsmTokenizer.from_pretrained(MODEL_DIR)
        _model = EsmModel.from_pretrained(MODEL_DIR)
    else:
        print("  Loading ESM-2 model (downloading ~1.2GB)...")
        _tokenizer = EsmTokenizer.from_pretrained(MODEL_NAME)
        _model = EsmModel.from_pretrained(MODEL_NAME)
    _model.eval()
    print("  ESM-2 model loaded.")
    return _model, _tokenizer


def _load_mlm_model():
    global _mlm_model, _tokenizer
    if _mlm_model is not None:
        return _mlm_model, _tokenizer

    if _tokenizer is None:
        _load_model()

    if MODEL_DIR.exists():
        print("  Loading ESM-2 MLM head for probability computation...")
        _mlm_model = EsmForMaskedLM.from_pretrained(MODEL_DIR)
    else:
        print("  Loading ESM-2 MLM model...")
        _mlm_model = EsmForMaskedLM.from_pretrained(MODEL_NAME)
    _mlm_model.eval()
    print("  ESM-2 MLM model loaded.")
    return _mlm_model, _tokenizer


def _get_aa_token_ids(tokenizer) -> List[int]:
    return [tokenizer.convert_tokens_to_ids(aa) for aa in AA_TOKENS]


def compute_residue_probabilities(fasta_text: str, max_length: int = 1022) -> List[Dict]:
    mlm_model, tokenizer = _load_mlm_model()
    entries = _parse_fasta(fasta_text)
    aa_token_ids = _get_aa_token_ids(tokenizer)
    num_aa = len(aa_token_ids)
    max_entropy = math.log(num_aa)

    all_probs = []
    with torch.no_grad():
        for seq_id, sequence in entries:
            seq_len = len(sequence)
            if seq_len == 0:
                all_probs.append({
                    "id": seq_id,
                    "sequence": sequence,
                    "p_true": [],
                    "entropy": [],
                    "conservation_score": [],
                })
                continue

            chunks = []
            for start in range(0, seq_len, max_length):
                chunk = sequence[start:start + max_length]
                chunks.append((start, chunk))

            full_p_true = torch.zeros(seq_len)
            full_entropy = torch.zeros(seq_len)
            full_conservation = torch.zeros(seq_len)

            for start, chunk in chunks:
                inputs = tokenizer(chunk, return_tensors="pt", padding=False, truncation=True, max_length=max_length)
                outputs = mlm_model(**inputs)
                logits = outputs.logits[0, 1:-1, :]

                aa_logits = logits[:, aa_token_ids]
                aa_probs = F.softmax(aa_logits, dim=-1)

                chunk_len = min(len(chunk), seq_len - start)
                actual_tokens = tokenizer(chunk, return_tensors="pt")["input_ids"][0, 1:1 + chunk_len]

                for i in range(chunk_len):
                    token_id = actual_tokens[i].item()
                    try:
                        aa_idx = aa_token_ids.index(token_id)
                        p_true = aa_probs[i, aa_idx].item()
                    except ValueError:
                        p_true = 0.0

                    probs = aa_probs[i]
                    entropy = -(probs * torch.log(probs + 1e-10)).sum().item()
                    conservation = 1.0 - (entropy / max_entropy) if max_entropy > 0 else 0.0

                    full_p_true[start + i] = p_true
                    full_entropy[start + i] = entropy
                    full_conservation[start + i] = conservation

            all_probs.append({
                "id": seq_id,
                "sequence": sequence,
                "p_true": full_p_true.tolist(),
                "entropy": full_entropy.tolist(),
                "conservation_score": full_conservation.tolist(),
            })

    return all_probs


def _parse_fasta(fasta_text: str) -> List[Tuple[str, str]]:
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
    return entries


def compute_residue_embeddings(fasta_text: str, max_length: int = 1022) -> List[Dict]:
    model, tokenizer = _load_model()
    entries = _parse_fasta(fasta_text)

    all_embeddings = []
    with torch.no_grad():
        for seq_id, sequence in entries:
            seq_len = len(sequence)
            if seq_len == 0:
                all_embeddings.append({"id": seq_id, "sequence": sequence, "embeddings": []})
                continue

            chunks = []
            for start in range(0, seq_len, max_length):
                chunk = sequence[start:start + max_length]
                chunks.append((start, chunk))

            full_embeddings = torch.zeros(seq_len, model.config.hidden_size)
            for start, chunk in chunks:
                inputs = tokenizer(chunk, return_tensors="pt", padding=False, truncation=True, max_length=max_length)
                outputs = model(**inputs)
                residue_embs = outputs.last_hidden_state[0, 1:-1, :]
                end = min(start + max_length, seq_len)
                full_embeddings[start:end] = residue_embs[:end - start]

            all_embeddings.append({
                "id": seq_id,
                "sequence": sequence,
                "embeddings": full_embeddings,
            })

    return all_embeddings


def compute_sequence_embedding(fasta_text: str) -> List[Dict]:
    residue_data = compute_residue_embeddings(fasta_text)
    results = []
    for entry in residue_data:
        if entry["embeddings"].numel() == 0:
            results.append({"id": entry["id"], "sequence": entry["sequence"], "embedding": None})
            continue
        mean_emb = entry["embeddings"].mean(dim=0)
        norm = torch.norm(mean_emb)
        if norm > 0:
            mean_emb = mean_emb / norm
        results.append({"id": entry["id"], "sequence": entry["sequence"], "embedding": mean_emb})
    return results


def cosine_similarity_matrix(embs1: torch.Tensor, embs2: torch.Tensor) -> torch.Tensor:
    norm1 = embs1 / (embs1.norm(dim=1, keepdim=True) + 1e-8)
    norm2 = embs2 / (embs2.norm(dim=1, keepdim=True) + 1e-8)
    return torch.mm(norm1, norm2.t())


def embedding_distance(seq1: str, seq2: str) -> float:
    fasta = f">seq1\n{seq1}\n>seq2\n{seq2}"
    seqs = compute_sequence_embedding(fasta)
    if seqs[0]["embedding"] is None or seqs[1]["embedding"] is None:
        return 0.0
    sim = torch.dot(seqs[0]["embedding"], seqs[1]["embedding"]).item()
    return float(max(0.0, sim))


def is_protein_sequence(fasta_text: str) -> bool:
    amino_acids = set("ACDEFGHIKLMNPQRSTVWYXBZJUO")
    entries = _parse_fasta(fasta_text)
    if not entries:
        return False
    for _, seq in entries:
        non_gap = set(seq.replace("-", "").replace(".", ""))
        if non_gap and len(non_gap & amino_acids) / len(non_gap) > 0.5:
            return True
    return False
