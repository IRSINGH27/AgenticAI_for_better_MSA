import json
import pathlib
import math
from collections import Counter
from typing import Dict, List, Tuple, Optional

import numpy as np
from scipy import sparse


SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
KNOWLEDGE_DIR = SCRIPT_DIR.parent / "knowledge"
INDEX_PREFIX = "tfidf_index"


def get_kmers(sequence: str, k: int = 6) -> List[str]:
    seq = sequence.upper().replace(" ", "").replace("\n", "").replace(">", "")
    if len(seq) < k:
        return [seq]
    return [seq[i:i+k] for i in range(len(seq) - k + 1)]


def build_vocab(experiments: List[Dict], k: int = 6) -> Dict[str, int]:
    vocab = {}
    for exp in experiments:
        embedding = exp.get("tfidf_embedding", {})
        if embedding:
            for term in embedding.keys():
                if term not in vocab:
                    vocab[term] = len(vocab)
    return vocab


def dict_to_sparse_vector(embedding: Dict[str, float], vocab: Dict[str, int]) -> sparse.csr_matrix:
    if not embedding:
        return sparse.csr_matrix((1, len(vocab)))
    rows, cols, data = [], [], []
    for term, value in embedding.items():
        if term in vocab:
            rows.append(0)
            cols.append(vocab[term])
            data.append(float(value))
    return sparse.csr_matrix((data, (rows, cols)), shape=(1, len(vocab)))


def build_sparse_index(experiments: List[Dict], k: int = 6) -> Tuple[sparse.csr_matrix, Dict[str, int], List[str]]:
    vocab = build_vocab(experiments, k)
    if not vocab:
        return sparse.csr_matrix((0, 0)), {}, []

    experiment_ids = []
    rows_final = []
    cols_final = []
    data_final = []

    for row_idx, exp in enumerate(experiments):
        exp_id = exp.get("experiment_id", f"row_{row_idx}")
        experiment_ids.append(exp_id)
        embedding = exp.get("tfidf_embedding", {})
        for term, value in embedding.items():
            if term in vocab:
                rows_final.append(row_idx)
                cols_final.append(vocab[term])
                data_final.append(float(value))

    matrix = sparse.csr_matrix(
        (data_final, (rows_final, cols_final)),
        shape=(len(experiments), len(vocab))
    )

    return matrix, vocab, experiment_ids


def compute_query_vector(fasta_text: str, vocab: Dict[str, int], k: int = 6) -> sparse.csr_matrix:
    from embeddings import compute_tfidf_embedding
    embedding = compute_tfidf_embedding(fasta_text, k)
    return dict_to_sparse_vector(embedding, vocab)


def query_sparse(
    query_vec: sparse.csr_matrix,
    matrix: sparse.csr_matrix,
    experiment_ids: List[str],
    top_k: int = 3
) -> List[Tuple[str, float]]:
    if matrix.shape[0] == 0 or query_vec.nnz == 0:
        return []

    query_dense = query_vec.toarray().flatten()
    query_norm = np.linalg.norm(query_dense)
    if query_norm == 0:
        return []

    row_sq = matrix.multiply(matrix).sum(axis=1)
    matrix_norms = np.sqrt(np.asarray(row_sq).flatten())
    matrix_norms[matrix_norms == 0] = 1.0

    similarities = np.asarray(matrix @ query_dense).flatten() / (matrix_norms * query_norm)

    top_indices = np.argsort(similarities)[::-1][:top_k]
    results = []
    for idx in top_indices:
        sim = float(similarities[idx])
        if sim > 0.1:
            if idx < len(experiment_ids):
                results.append((experiment_ids[idx], sim))
    return results


def _resolve_path_prefix(path_prefix: Optional[pathlib.Path]) -> pathlib.Path:
    if path_prefix is None:
        return KNOWLEDGE_DIR / INDEX_PREFIX
    p = pathlib.Path(path_prefix)
    if p.is_dir() or str(p).endswith('/') or str(p) == '.':
        return pathlib.Path(str(p).rstrip('/')) / INDEX_PREFIX
    return p


def save_index(
    matrix: sparse.csr_matrix,
    vocab: Dict[str, int],
    experiment_ids: List[str],
    path_prefix: Optional[pathlib.Path] = None
) -> None:
    path_prefix = _resolve_path_prefix(path_prefix)
    KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
    sparse.save_npz(str(path_prefix) + ".npz", matrix)

    vocab_data = {
        "vocab": vocab,
        "experiment_ids": experiment_ids
    }
    with open(str(path_prefix) + "_vocab.json", "w") as f:
        json.dump(vocab_data, f, indent=2)


def load_index(path_prefix: Optional[pathlib.Path] = None) -> Tuple[sparse.csr_matrix, Dict[str, int], List[str]]:
    path_prefix = _resolve_path_prefix(path_prefix)
    npz_path = pathlib.Path(str(path_prefix) + ".npz")
    vocab_path = pathlib.Path(str(path_prefix) + "_vocab.json")

    if not npz_path.exists() or not vocab_path.exists():
        return sparse.csr_matrix((0, 0)), {}, []

    matrix = sparse.load_npz(str(npz_path))
    with open(vocab_path, "r") as f:
        vocab_data = json.load(f)

    return matrix, vocab_data.get("vocab", {}), vocab_data.get("experiment_ids", [])


def add_to_index(
    matrix: sparse.csr_matrix,
    vocab: Dict[str, int],
    experiment_ids: List[str],
    new_id: str,
    new_embedding: Dict[str, float]
) -> Tuple[sparse.csr_matrix, Dict[str, int], List[str]]:
    new_terms = set(new_embedding.keys()) - set(vocab.keys())
    if new_terms:
        for term in new_terms:
            vocab[term] = len(vocab)
        old_n_cols = matrix.shape[1]
        new_n_cols = len(vocab)
        if new_n_cols > old_n_cols:
            matrix = sparse.hstack([matrix, sparse.csr_matrix((matrix.shape[0], new_n_cols - old_n_cols))])
            matrix = matrix.tocsr()

    new_row = dict_to_sparse_vector(new_embedding, vocab)
    matrix = sparse.vstack([matrix, new_row]).tocsr()
    experiment_ids.append(new_id)
    return matrix, vocab, experiment_ids


def get_index_stats(matrix: sparse.csr_matrix, vocab: Dict[str, int]) -> Dict:
    if matrix.shape[0] == 0:
        return {
            "num_documents": 0,
            "vocab_size": 0,
            "matrix_shape": (0, 0),
            "nnz": 0,
            "density": 0.0,
            "avg_terms_per_doc": 0.0
        }

    nnz = matrix.nnz
    total_cells = matrix.shape[0] * matrix.shape[1]
    density = nnz / total_cells if total_cells > 0 else 0.0
    avg_terms = nnz / matrix.shape[0] if matrix.shape[0] > 0 else 0.0

    return {
        "num_documents": matrix.shape[0],
        "vocab_size": matrix.shape[1],
        "matrix_shape": matrix.shape,
        "nnz": nnz,
        "density": round(density, 6),
        "avg_terms_per_doc": round(avg_terms, 2)
    }
