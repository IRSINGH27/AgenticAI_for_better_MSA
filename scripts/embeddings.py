import math
import re
from collections import Counter
from typing import Dict, List, Set, Tuple


def get_kmers(sequence: str, k: int = 6) -> List[str]:
    seq = sequence.upper().replace(" ", "").replace("\n", "").replace(">", "")
    if len(seq) < k:
        return [seq]
    return [seq[i:i+k] for i in range(len(seq) - k + 1)]


def compute_tfidf_embedding(fasta_text: str, k: int = 6) -> Dict[str, float]:
    sequences = []
    current_seq = []

    for line in fasta_text.strip().splitlines():
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

    if not sequences:
        return {}

    all_kmers = []
    seq_kmers_list = []

    for seq in sequences:
        kmers = get_kmers(seq, k)
        seq_kmers_list.append(kmers)
        all_kmers.extend(kmers)

    if not all_kmers:
        return {}

    seq_kmer_sets = [set(kmers) for kmers in seq_kmers_list]
    doc_freq = {kmer: sum(1 for s in seq_kmer_sets if kmer in s) for kmer in set(all_kmers)}
    num_docs = len(sequences)
    idf = {kmer: math.log((num_docs + 1) / (freq + 1)) + 1 for kmer, freq in doc_freq.items()}

    tfidf_vector = {}
    for kmers in seq_kmers_list:
        kmer_counts = Counter(kmers)
        total_kmers = len(kmers) if kmers else 1
        for kmer, count in kmer_counts.items():
            tf = count / total_kmers
            tfidf = tf * idf.get(kmer, 0)
            tfidf_vector[kmer] = tfidf_vector.get(kmer, 0) + tfidf

    if tfidf_vector:
        norm = math.sqrt(sum(v ** 2 for v in tfidf_vector.values()))
        if norm > 0:
            tfidf_vector = {k: v / norm for k, v in tfidf_vector.items()}

    return tfidf_vector


def cosine_similarity(vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
    if not vec1 or not vec2:
        return 0.0

    common_keys = set(vec1.keys()) & set(vec2.keys())
    if not common_keys:
        return 0.0

    dot_product = sum(vec1[k] * vec2[k] for k in common_keys)
    norm1 = math.sqrt(sum(v ** 2 for v in vec1.values()))
    norm2 = math.sqrt(sum(v ** 2 for v in vec2.values()))

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def find_similar_experiments(
    fasta_text: str,
    experiments: List[Dict],
    top_k: int = 3,
    k: int = 6
) -> List[Tuple[float, Dict]]:
    current_vec = compute_tfidf_embedding(fasta_text, k)

    if not current_vec:
        return []

    scored = []
    for exp in experiments:
        if not exp.get("tfidf_embedding"):
            continue

        stored_vec = exp["tfidf_embedding"]
        similarity = cosine_similarity(current_vec, stored_vec)

        if similarity > 0.1:
            scored.append((similarity, exp))

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_k]


def get_sequence_fingerprint(fasta_text: str) -> Dict:
    sequences = []
    current_seq = []

    for line in fasta_text.strip().splitlines():
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

    num_seqs = len(sequences)
    total_length = sum(len(s) for s in sequences)
    avg_length = total_length / num_seqs if num_seqs > 0 else 0

    all_chars = "".join(sequences).upper()
    char_counts = Counter(all_chars)
    total_chars = sum(char_counts.values())

    char_freq = {char: count / total_chars for char, count in char_counts.items()} if total_chars > 0 else {}

    nucleotide_chars = set("ACGTUN")
    protein_chars = set("ACDEFGHIKLMNPQRSTVWY")

    is_nucleotide = sum(char_freq.get(c, 0) for c in nucleotide_chars) > 0.7
    is_protein = sum(char_freq.get(c, 0) for c in protein_chars) > 0.7

    seq_type = "nucleotide" if is_nucleotide else "protein" if is_protein else "unknown"

    return {
        "num_sequences": num_seqs,
        "total_length": total_length,
        "avg_length": round(avg_length, 1),
        "seq_type": seq_type,
        "char_freq": char_freq
    }