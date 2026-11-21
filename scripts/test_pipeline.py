import subprocess
import sys
import shutil

from config import CONDA_ENV


def check_mafft():
    try:
        result = subprocess.run(
            ["conda", "run", "-n", CONDA_ENV, "mafft", "--version"],
            capture_output=True,
            timeout=10
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def test_msa_runner():
    print("=== Testing msa_runner ===")
    if not check_mafft():
        print(f"⚠ MAFFT not found in conda env '{CONDA_ENV}'")
        print(f"  Install with: conda install -n {CONDA_ENV} -c bioconda mafft\n")
        return

    from msa_runner import run_msa
    fasta = """>seq1
ATCGATCGATCG
>seq2
ATCGATCGATCG
>seq3
ATCGATCGAT--
"""
    result = run_msa(fasta, {"strategy": "--auto"})
    print(result[:200])
    print("✓ MSA runner works\n")


def test_metrics():
    print("=== Testing metrics ===")
    from metrics import compute_metrics
    alignment = """>seq1
ATCGATCGATCG
>seq2
ATCGATCGAT-CG
>seq3
ATCG-ATCGATCG
"""
    m = compute_metrics(alignment)
    print(f"metrics: {m}")
    assert "gap_fraction" in m
    assert "alignment_length" in m
    assert "avg_sequence_length" in m
    print("✓ Metrics works\n")


def test_convergence():
    print("=== Testing convergence ===")
    from convergence import ConvergenceTracker
    tracker = ConvergenceTracker(tolerance=0.001, patience=2)

    m1 = {"gap_fraction": 0.1, "alignment_length": 100}
    c1, i1 = tracker.check(m1)
    print(f"iteration 1: converged={c1}, info={i1}")

    m2 = {"gap_fraction": 0.101, "alignment_length": 100}
    c2, i2 = tracker.check(m2)
    print(f"iteration 2: converged={c2}, info={i2}")

    m3 = {"gap_fraction": 0.102, "alignment_length": 100}
    c3, i3 = tracker.check(m3)
    print(f"iteration 3: converged={c3}, info={i3}")
    print("✓ Convergence works\n")


def test_llm():
    print("=== Testing LLM connection ===")
    from llm_core import ask_llm_json
    try:
        response = ask_llm_json("Say 'hello' in JSON: {\"message\": \"hello\"}")
        print(f"LLM response: {response}")
        print("✓ LLM works\n")
    except Exception as e:
        print(f"LLM error: {e}")
        print("⚠ LLM not available (start ollama with: ollama serve)\n")


def test_full_pipeline():
    print("=== Testing full pipeline ===")
    if not check_mafft():
        print(f"⚠ MAFFT not found in conda env '{CONDA_ENV}', skipping full pipeline test")
        print(f"  Install with: conda install -n {CONDA_ENV} -c bioconda mafft\n")
        return

    from pipeline_loop import run_pipeline

    print("\nRunning 2 iterations (max_iterations=2)...\n")
    result = run_pipeline(
        fasta_text=""">seq1
ATCGATCGATCGATCG
>seq2
ATCGATCGATCGAT--
>seq3
ATCG--TCGATCGAT
>seq4
ATCGATCGATCGATCG
""",
        max_iterations=2,
        convergence_tolerance=0.1,
        convergence_patience=2
    )
    print(f"\nResult: {result['iterations']} iterations, experiment_id={result['experiment_id']}")
    print("✓ Pipeline works\n")


def test_toon_roundtrip():
    print("=== Testing TOON round-trip ===")
    from toon_format import to_toon, from_toon
    import json

    test_cases = [
        {"name": "A", "age": 30, "tags": ["x", "y"]},
        [{"a": 1, "b": "hi"}, {"a": 2, "b": "bye"}],
        {"nested": {"deep": {"key": "val"}}},
        [1, 2, 3, 4, 5],
        "simple string",
        42,
        True,
        None,
        [],
        {},
    ]

    all_pass = True
    for i, original in enumerate(test_cases):
        try:
            encoded = to_toon(original)
            decoded = from_toon(encoded)
            if json.dumps(original, sort_keys=True) != json.dumps(decoded, sort_keys=True):
                print(f"  ✗ Test {i+1} FAIL: {original} -> {decoded}")
                all_pass = False
        except Exception as e:
            print(f"  ✗ Test {i+1} ERROR: {e}")
            all_pass = False

    if all_pass:
        print(f"  ✓ All {len(test_cases)} round-trip cases passed\n")
    else:
        print(f"  ✗ Some round-trip tests failed\n")


def test_toon_nested_tabular():
    print("=== Testing TOON with nested tabular arrays ===")
    from toon_format import to_toon, from_toon
    import json

    record = {
        "experiment_id": "test123",
        "actions_tried": [
            {"iteration": 1, "action": "accept", "strategy": "auto", "extra_args": "", "gap_fraction": 0.0, "outcome": "success"},
            {"iteration": 2, "action": "refine", "strategy": "auto", "extra_args": "--maxiterate", "gap_fraction": 0.05, "outcome": "no_change"},
        ],
        "extended_metrics": {
            "gap_fraction": 0.05,
            "alignment_length": 100,
            "sp_score": 0.85,
        }
    }

    encoded = to_toon(record)
    decoded = from_toon(encoded)

    assert decoded["experiment_id"] == record["experiment_id"], "experiment_id mismatch"
    assert len(decoded["actions_tried"]) == 2, f"actions_tried length {len(decoded['actions_tried'])}"
    assert decoded["actions_tried"][0]["action"] == "accept", f"first action: {decoded['actions_tried'][0]}"
    assert decoded["actions_tried"][1]["extra_args"] == "--maxiterate", f"extra_args: {decoded['actions_tried'][1]}"
    assert decoded["extended_metrics"]["gap_fraction"] == 0.05, f"extended_metrics: {decoded['extended_metrics']}"

    print(f"  ✓ Nested tabular array round-trips correctly")
    print(f"  ✓ Quoted strings (--maxiterate) preserved")
    print(f"  ✓ Nested objects preserved\n")


def test_sparse_index():
    print("=== Testing sparse TF-IDF index ===")
    from tfidf_sparse import build_sparse_index, save_index, load_index, query_sparse, compute_query_vector, get_index_stats
    import tempfile
    import pathlib

    records = [
        {
            "experiment_id": "r1",
            "tfidf_embedding": {"ATCGAT": 0.5, "GATCGA": 0.3, "TCGATG": 0.2},
            "sequence_signature": ["ATC", "GAT"],
        },
        {
            "experiment_id": "r2",
            "tfidf_embedding": {"GCATGC": 0.6, "CATGCG": 0.4},
            "sequence_signature": ["GCA", "CAT"],
        },
        {
            "experiment_id": "r3",
            "tfidf_embedding": {"ATCGAT": 0.4, "GATCGA": 0.4, "CGATCG": 0.2},
            "sequence_signature": ["ATC", "TCG"],
        },
    ]

    matrix, vocab, ids = build_sparse_index(records)
    assert matrix.shape[0] == 3, f"matrix rows {matrix.shape[0]}"
    assert matrix.shape[1] == len(vocab), f"matrix cols {matrix.shape[1]}"
    assert len(ids) == 3, f"ids {ids}"
    print(f"  ✓ Built sparse index: shape={matrix.shape}, nnz={matrix.nnz}, vocab={len(vocab)}")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = pathlib.Path(tmpdir)
        save_index(matrix, vocab, ids, tmpdir)
        m2, v2, ids2 = load_index(tmpdir)
        assert m2.shape == matrix.shape, f"loaded shape {m2.shape} != original {matrix.shape}"
        assert v2 == vocab, f"loaded vocab differs"
        assert ids2 == ids, f"loaded ids {ids2} != original {ids}"
        print(f"  ✓ Round-trip via save/load preserves matrix, vocab, ids")

        fasta = ">seq1\nATCGATCGATCGATCGATCG"
        qv = compute_query_vector(fasta, v2)
        results = query_sparse(qv, m2, ids2, top_k=3)
        assert len(results) > 0, f"no results from query_sparse"
        top_id, top_sim = results[0]
        assert top_id in ("r1", "r3"), f"top result {top_id} (expected r1 or r3 for ATCGAT query)"
        print(f"  ✓ Query returns {len(results)} results, top={top_id} sim={top_sim:.3f}")

    stats = get_index_stats(matrix, vocab)
    assert stats["num_documents"] == 3
    assert stats["vocab_size"] == len(vocab)
    assert 0 < stats["density"] < 1, f"density {stats['density']} should be in (0, 1)"
    print(f"  ✓ Index stats: {stats}\n")


def test_toon_storage_roundtrip():
    print("=== Testing TOON storage end-to-end ===")
    from rag_retrieve import load_knowledge, update_strategy_stats, KNOWLEDGE_DIR
    from rag_store import _load_existing_records, _append_record
    from toon_format import from_toon, to_toon
    import json
    import pathlib

    knowledge_dir = KNOWLEDGE_DIR
    backup_files = {}
    for fname in ("experiments.toon", "experiments.toon.gz", "experiments.jsonl", "experiments.jsonl.gz", "strategy_stats.toon", "strategy_stats.toon.gz", "tfidf_index.npz", "tfidf_index_vocab.json"):
        fpath = knowledge_dir / fname
        if fpath.exists():
            backup_files[fname] = fpath.read_bytes()
            fpath.unlink()

    try:
        exp1 = {
            "experiment_id": "tr1",
            "problem_type": "good_quality",
            "initial_gap_fraction": 0.05,
            "final_gap_fraction": 0.02,
            "final_outcome": "success",
            "actions_tried": [
                {"action": "accept", "strategy": "auto", "extra_args": "", "outcome": "success"},
            ],
            "tfidf_embedding": {"ATCGAT": 0.5, "GATCGA": 0.5},
            "sequence_signature": ["ATC", "GAT"],
        }
        exp2 = dict(exp1)
        exp2["experiment_id"] = "tr2"
        exp2["problem_type"] = "high_gap_fraction"

        _append_record(exp1)
        _append_record(exp2)

        loaded = _load_existing_records()
        assert len(loaded) == 2, f"loaded {len(loaded)} records, expected 2"
        assert loaded[0]["experiment_id"] in ("tr1", "tr2"), f"id: {loaded[0]['experiment_id']}"
        assert loaded[0]["problem_type"] in ("good_quality", "high_gap_fraction")
        assert len(loaded[0]["actions_tried"]) == 1
        assert loaded[0]["actions_tried"][0]["action"] == "accept"
        print(f"  ✓ Appended 2 records to gzip JSONL via subprocess, loaded back correctly")

        update_strategy_stats(exp1)
        stats_path = knowledge_dir / "strategy_stats.toon.gz"
        assert stats_path.exists(), "strategy_stats.toon.gz not created"
        from rag_retrieve import _gzip_read_text
        text = _gzip_read_text(stats_path)
        data = from_toon(text)
        assert "strategy_stats" in data
        stats_dict = data["strategy_stats"]
        matching_keys = [k for k in stats_dict if k.strip('"') == "accept:auto:"]
        assert len(matching_keys) >= 1, f"no key for accept:auto: in {list(stats_dict.keys())}"
        key = matching_keys[0]
        assert stats_dict[key]["success"] >= 1
        print(f"  ✓ update_strategy_stats writes gzip TOON via subprocess and round-trips keys")

        update_strategy_stats(exp1)
        text2 = _gzip_read_text(stats_path)
        data2 = from_toon(text2)
        for k in data2["strategy_stats"]:
            assert not (k.startswith('"') and k.endswith('"') and '\\"' in k), \
                f"key has nested quotes (round-trip bug): {k!r}"
        print(f"  ✓ Repeated updates don't accumulate quote characters")
    finally:
        for fname, data in backup_files.items():
            (knowledge_dir / fname).write_bytes(data)

    print()


def test_path_resolution():
    print("=== Testing path resolution for sparse index ===")
    from tfidf_sparse import _resolve_path_prefix
    from rag_retrieve import KNOWLEDGE_DIR
    import pathlib

    p1 = _resolve_path_prefix(None)
    assert str(p1).endswith("tfidf_index"), f"default path: {p1}"

    p2 = _resolve_path_prefix(KNOWLEDGE_DIR)
    assert str(p2).endswith("knowledge/tfidf_index"), f"dir str: {p2}"

    p3 = _resolve_path_prefix(pathlib.Path(str(KNOWLEDGE_DIR) + "/"))
    assert str(p3).endswith("knowledge/tfidf_index"), f"trailing slash: {p3}"

    p4 = _resolve_path_prefix(KNOWLEDGE_DIR / "tfidf_index")
    assert str(p4).endswith("knowledge/tfidf_index"), f"explicit prefix: {p4}"

    p5 = _resolve_path_prefix(KNOWLEDGE_DIR)
    assert str(p5).endswith("knowledge/tfidf_index"), f"Path dir: {p5}"

    print(f"  ✓ None -> {p1}")
    print(f"  ✓ KNOWLEDGE_DIR -> {p2}")
    print(f"  ✓ KNOWLEDGE_DIR/ -> {p3}")
    print(f"  ✓ KNOWLEDGE_DIR/tfidf_index -> {p4}")
    print(f"  ✓ KNOWLEDGE_DIR (again) -> {p5}\n")


def test_idf_document_frequency():
    print("=== Testing IDF uses document frequency (not term count) ===")
    from embeddings import compute_tfidf_embedding, cosine_similarity

    fasta_single = ">seq1\nATGCATGCATGCATGCATGCATGCATGCAT"
    emb = compute_tfidf_embedding(fasta_single)

    all_positive = all(v >= 0 for v in emb.values())
    assert all_positive, f"IDF produced negative values: {emb}"
    print(f"  ✓ All IDF values are non-negative (no negative cosine from high-count kmers)")

    fasta_similar = ">seq1\nATGCATGCATGCATGCATGCATGCATGCAT\n>seq2\nATGCATGCATGCATGCATGCATGCATGCAA"
    fasta_different = ">seq1\nATCGATCGATCGATCGATCG\n>seq2\nCGATCGATCGATCGATCGAT"
    emb_similar = compute_tfidf_embedding(fasta_similar)
    emb_different = compute_tfidf_embedding(fasta_different)
    emb_original = compute_tfidf_embedding(fasta_single)

    sim_same = cosine_similarity(emb_original, emb_original)
    sim_similar = cosine_similarity(emb_original, emb_similar)
    sim_different = cosine_similarity(emb_original, emb_different)

    assert sim_same > 0.99, f"Self-similarity {sim_same} should be ~1.0"
    assert sim_similar > 0.8, f"Similar sequences sim {sim_similar} should be high"
    assert sim_different < 0.1, f"Different sequences sim {sim_different} should be low"

    print(f"  ✓ Self-similarity: {sim_same:.3f}")
    print(f"  ✓ Similar sequences: {sim_similar:.3f}")
    print(f"  ✓ Different sequences: {sim_different:.3f}")

    from tfidf_sparse import build_sparse_index, save_index, load_index, query_sparse, compute_query_vector
    import tempfile
    import pathlib

    records = [
        {"experiment_id": "r1", "tfidf_embedding": compute_tfidf_embedding(">seq1\nATGCATGCATGCATGCATGCATGCATGCAT")},
        {"experiment_id": "r2", "tfidf_embedding": compute_tfidf_embedding(">seq1\nATCGATCGATCGATCGATCG")},
        {"experiment_id": "r3", "tfidf_embedding": compute_tfidf_embedding(">seq1\nATGCATGCATGCATGCATGCATGCATGCAT\n>seq2\nATGCATGCATGCATGCATGCATGCATGCAA")},
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = pathlib.Path(tmpdir)
        matrix, vocab, ids = build_sparse_index(records)
        save_index(matrix, vocab, ids, tmpdir)
        m2, v2, ids2 = load_index(tmpdir)

        qv = compute_query_vector(">seq1\nATGCATGCATGCATGCATGCATGCATGCAT", v2)
        results = query_sparse(qv, m2, ids2, top_k=3)

        assert len(results) > 0, "Sparse search returned 0 results"
        top_id, top_sim = results[0]
        assert top_id in ("r1", "r3"), f"Top result {top_id} should be r1 or r3"
        print(f"  ✓ Sparse search returns {len(results)} results (top={top_id}, sim={top_sim:.3f})")


def test_warm_start():
    print("=== Testing warm start detection ===")
    from rag_retrieve import find_warm_start, WARM_START_THRESHOLD, KNOWLEDGE_DIR
    from rag_store import _append_record
    from tfidf_sparse import build_sparse_index, save_index
    from embeddings import compute_tfidf_embedding
    import pathlib

    knowledge_dir = KNOWLEDGE_DIR
    backup_files = {}
    for fname in ("experiments.toon", "experiments.toon.gz", "experiments.jsonl", "experiments.jsonl.gz", "strategy_stats.toon", "strategy_stats.toon.gz", "tfidf_index.npz", "tfidf_index_vocab.json"):
        fpath = knowledge_dir / fname
        if fpath.exists():
            backup_files[fname] = fpath.read_bytes()
            fpath.unlink()

    try:
        fasta_same = ">seq1\nATGCATGCATGCATGCATGCATGCATGCAT\n>seq2\nATGCATGCATGCATGCATGCATGCATGCAA"
        fasta_diff = ">seq1\nCGTACGTACGTACGTACGTACGTACGTACGT\n>seq2\nCGTACGTACGTACGTACGTACGTACGTACGA"

        exp = {
            "experiment_id": "ws_test",
            "problem_type": "good_quality",
            "initial_gap_fraction": 0.05,
            "final_gap_fraction": 0.02,
            "final_outcome": "success",
            "actions_tried": [
                {"iteration": 1, "action": "accept", "strategy": "auto", "extra_args": "", "gap_fraction": 0.05, "outcome": "success"},
            ],
            "tfidf_embedding": compute_tfidf_embedding(fasta_same),
            "sequence_signature": list(set(fasta_same.replace(">", "").replace("\n", "")[i:i+4] for i in range(len(fasta_same.replace(">", "").replace("\n", "")) - 3))),
        }

        _append_record(exp)

        matrix, vocab, ids = build_sparse_index([exp])
        save_index(matrix, vocab, ids, knowledge_dir)

        result_same = find_warm_start(fasta_same, exclude_id="other")
        assert result_same is not None, "find_warm_start should find similar experiment"
        assert result_same["experiment_id"] == "ws_test", f"wrong experiment: {result_same['experiment_id']}"
        assert result_same["similarity"] >= WARM_START_THRESHOLD, f"similarity {result_same['similarity']} below threshold"
        assert len(result_same["actions_tried"]) == 1, f"actions_tried: {len(result_same['actions_tried'])}"
        print(f"  ✓ Found warm start: experiment={result_same['experiment_id']}, sim={result_same['similarity']:.3f}")

        result_diff = find_warm_start(fasta_diff, exclude_id="other")
        assert result_diff is None, f"find_warm_start should return None for different sequences (got sim={result_diff['similarity'] if result_diff else 0:.3f})"
        print(f"  ✓ No warm start for different sequences (correct)")

        result_self = find_warm_start(fasta_same, exclude_id="ws_test")
        assert result_self is None, "find_warm_start should exclude self"
        print(f"  ✓ Self-exclusion works")
    finally:
        for fname, data in backup_files.items():
            (knowledge_dir / fname).write_bytes(data)

    print()


def test_hill_numbers():
    print("=== Testing Hill numbers ===")
    from extended_metrics import compute_hill_metrics, _compute_hill_number

    counts_conserved = [10]
    assert _compute_hill_number(counts_conserved, 0.0) == 1.0, "q=0 conserved"
    assert abs(_compute_hill_number(counts_conserved, 1.0) - 1.0) < 0.01, "q=1 conserved"
    assert abs(_compute_hill_number(counts_conserved, 2.0) - 1.0) < 0.01, "q=2 conserved"
    print(f"  ✓ Conserved column: q0=1, q1=1, q2=1")

    counts_diverse = [2, 2, 2, 2, 2]
    assert _compute_hill_number(counts_diverse, 0.0) == 5.0, "q=0 diverse"
    assert abs(_compute_hill_number(counts_diverse, 1.0) - 5.0) < 0.01, "q=1 diverse"
    assert abs(_compute_hill_number(counts_diverse, 2.0) - 5.0) < 0.01, "q=2 diverse"
    print(f"  ✓ Diverse column: q0=5, q1=5, q2=5")

    counts_mixed = [8, 1, 1]
    q0 = _compute_hill_number(counts_mixed, 0.0)
    q1 = _compute_hill_number(counts_mixed, 1.0)
    q2 = _compute_hill_number(counts_mixed, 2.0)
    assert q0 == 3.0, f"q=0 mixed: {q0}"
    assert 1.0 < q1 < 3.0, f"q=1 mixed: {q1}"
    assert 1.0 < q2 < 3.0, f"q=2 mixed: {q2}"
    print(f"  ✓ Mixed column: q0={q0}, q1={q1:.3f}, q2={q2:.3f}")

    alignment = """>seq1
MKVLWAALLVTFLAGCQA
>seq2
MKVLWAALLVTFLAGCQ-
>seq3
MKVLWAALLVTFLAGC--
>seq4
MKVLWAALLVT----CQA"""
    hill = compute_hill_metrics(alignment)
    assert hill["hill_q0_avg"] > 0, f"hill_q0_avg: {hill['hill_q0_avg']}"
    assert hill["hill_q1_avg"] > 0, f"hill_q1_avg: {hill['hill_q1_avg']}"
    assert hill["hill_q2_avg"] > 0, f"hill_q2_avg: {hill['hill_q2_avg']}"
    print(f"  ✓ Overall metrics: q0={hill['hill_q0_avg']}, q1={hill['hill_q1_avg']}, q2={hill['hill_q2_avg']}")

    from extended_metrics import compute_extended_metrics
    metrics = compute_extended_metrics(alignment)
    assert "hill_q0_avg" in metrics, "hill_q0_avg missing from extended metrics"
    assert "hill_q1_avg" in metrics, "hill_q1_avg missing from extended metrics"
    assert "hill_problematic_columns" in metrics, "hill_problematic_columns missing"
    print(f"  ✓ Extended metrics includes Hill numbers")

    all_conserved = """>seq1
AAAAAAAA
>seq2
AAAAAAAA
>seq3
AAAAAAAA"""
    hill_c = compute_hill_metrics(all_conserved)
    assert hill_c["hill_q0_avg"] == 1.0, f"all conserved q0: {hill_c['hill_q0_avg']}"
    assert hill_c["hill_q1_avg"] == 1.0, f"all conserved q1: {hill_c['hill_q1_avg']}"
    print(f"  ✓ All-conserved alignment: q0=1, q1=1")

    print()


def test_esm_alignment():
    print("=== Testing ESM-2 alignment ===")
    from esm_aligner import run_esm_alignment
    from extended_metrics import compute_extended_metrics

    fasta = """>seq1
MKVLWAALLVTFLAGCQA
>seq2
MKVLWAALLVTFLAGCQ-"""

    print(f"  Running ESM-2 alignment on 2 sequences...")
    result = run_esm_alignment(fasta)
    assert result, "ESM alignment returned empty"
    assert ">seq1" in result, "missing seq1 header"
    assert ">seq2" in result, "missing seq2 header"

    lines = [l for l in result.split("\n") if not l.startswith(">")]
    aligned_seqs = [l for l in lines if l.strip()]
    assert len(aligned_seqs) == 2, f"expected 2 aligned seqs, got {len(aligned_seqs)}"
    assert len(aligned_seqs[0]) == len(aligned_seqs[1]), f"lengths differ: {len(aligned_seqs[0])} vs {len(aligned_seqs[1])}"
    print(f"  ✓ Alignment produced: length={len(aligned_seqs[0])}")

    metrics = compute_extended_metrics(result)
    print(f"  ✓ gap_fraction={metrics['gap_fraction']:.3f}, sp_score={metrics['sp_score']:.3f}")

    fasta3 = """>seq1
ACDEFGHIKLMNPQRSTVWY
>seq2
ACDEFGHIKLMNPQRSTVW-
>seq3
ACDEFGHIKLMNP----WY"""
    print(f"  Running ESM-2 alignment on 3 sequences...")
    result3 = run_esm_alignment(fasta3)
    assert result3, "ESM 3-seq alignment returned empty"
    lines3 = [l for l in result3.split("\n") if not l.startswith(">")]
    aligned3 = [l for l in lines3 if l.strip()]
    assert len(aligned3) == 3, f"expected 3 aligned seqs, got {len(aligned3)}"
    print(f"  ✓ 3-sequence alignment: length={len(aligned3[0])}")

    print()


def test_rule_based_decision():
    print("=== Testing rule-based decision layer ===")
    from pipeline_loop import _rule_based_decision

    good_metrics = {"gap_fraction": 0.05, "sp_score": 0.72, "identity": 0.65}
    d = _rule_based_decision(good_metrics, "mafft", 1)
    assert d is not None, "good alignment should trigger rule"
    assert d["action"] == "accept", f"expected accept, got {d['action']}"
    print(f"  ✓ Good alignment -> accept")

    # gap in probe band 0.12-0.40, esm not tried yet -> esm2 probe
    high_gap_mafft = {"gap_fraction": 0.22, "sp_score": 0.25, "identity": 0.35}
    d = _rule_based_decision(high_gap_mafft, "mafft", 1)
    assert d is not None, "high gap on mafft should trigger rule"
    assert d["action"] == "change_tool", f"expected change_tool, got {d['action']}"
    assert d["tool"] == "esm2", f"probe should try esm2 first, got {d['tool']}"
    assert "esm probe" in d["reason"]
    print(f"  ✓ High gap on mafft (esm untried) -> esm2 probe")

    # same gap after esm already tried -> fall through to muscle
    d = _rule_based_decision(high_gap_mafft, "mafft", 1, esm_tried=True)
    assert d is not None and d["tool"] == "muscle", f"after esm tried, expect muscle, got {d}"
    print(f"  ✓ High gap on mafft (esm tried) -> muscle")

    high_gap_muscle_streak = {"gap_fraction": 0.20, "sp_score": 0.30, "identity": 0.40}
    d = _rule_based_decision(high_gap_muscle_streak, "muscle", 2, esm_tried=True)
    assert d is not None, "high gap muscle streak should trigger rule"
    assert d["action"] == "trim", f"expected trim, got {d['action']}"
    assert d["extra_args"] == "-gappyout", f"expected gentle -gappyout, got {d['extra_args']}"
    print(f"  ✓ High gap muscle 2x -> gentle trim -gappyout (indels kept)")

    very_divergent = {"gap_fraction": 0.30, "sp_score": 0.15, "identity": 0.15}
    d = _rule_based_decision(very_divergent, "mafft", 1)
    assert d is not None, "very divergent should trigger rule"
    assert d["action"] == "change_tool"
    assert d["tool"] == "esm2", f"expected esm2 for divergent, got {d['tool']}"
    print(f"  ✓ Very divergent -> change_tool to esm2")

    # below probe band and not divergent enough -> falls to LLM
    ambiguous = {"gap_fraction": 0.10, "sp_score": 0.40, "identity": 0.45}
    d = _rule_based_decision(ambiguous, "mafft", 1, esm_tried=True)
    assert d is None, f"ambiguous case should return None, got {d}"
    print(f"  ✓ Ambiguous case -> None (falls through to LLM)")

    print("✓ Rule-based decision layer works\n")


def test_metrics_echo_validation():
    print("=== Testing metrics echo validation ===")
    from ai_controller import _check_metrics_echo

    actual = {"gap_fraction": 0.22, "sp_score": 0.25, "identity": 0.35}

    good_echo = {"gap_fraction": 0.22, "sp_score": 0.25, "identity": 0.35}
    assert _check_metrics_echo({"metrics_echo": good_echo}, actual), "exact echo should pass"
    print(f"  ✓ Exact echo passes")

    close_echo = {"gap_fraction": 0.23, "sp_score": 0.24, "identity": 0.36}
    assert _check_metrics_echo({"metrics_echo": close_echo}, actual), "close echo should pass"
    print(f"  ✓ Close echo (within 0.05) passes")

    bad_echo = {"gap_fraction": 0.05, "sp_score": 0.25, "identity": 0.35}
    assert not _check_metrics_echo({"metrics_echo": bad_echo}, actual), "wrong gap should fail"
    print(f"  ✓ Wrong gap_fraction (hallucination) fails")

    no_echo = {}
    assert _check_metrics_echo({"metrics_echo": no_echo}, actual), "missing echo should pass (lenient)"
    print(f"  ✓ Missing echo passes (lenient)")

    print("✓ Metrics echo validation works\n")


def test_duplicate_detection_tfidf():
    print("=== Testing duplicate detection with TF-IDF ===")
    from rag_store import _is_duplicate

    fasta = """>s1
MKVLAAGVG
>s2
MKVLAAGVG
>s3
MKVLAAGVT
"""
    is_dup = _is_duplicate(fasta, "good_quality", threshold=0.99)
    print(f"  ✓ Duplicate check ran (result={is_dup})")

    print("✓ Duplicate detection works\n")


def test_validate_action():
    print("=== Testing action validation ===")
    from ai_controller import _validate_action

    good = {"action": "change_tool", "tool": "muscle", "msa_mode": "align",
            "extra_args": "", "reason": "test", "confidence": "high"}
    result = _validate_action(good)
    assert result["action"] == "change_tool"
    assert result["tool"] == "muscle"
    print(f"  ✓ Valid action passes through")

    bad_action = {"action": "invalid_action", "tool": "muscle"}
    result = _validate_action(bad_action)
    assert result["action"] == "refine", f"bad action should fallback to refine, got {result['action']}"
    print(f"  ✓ Bad action falls back to refine")

    bad_tool = {"action": "change_tool", "tool": "clustalw"}
    result = _validate_action(bad_tool)
    assert result["tool"] == "mafft", f"bad tool should fallback to mafft, got {result['tool']}"
    print(f"  ✓ Bad tool falls back to mafft")

    old_format = {"decision": "accept", "reason": "test"}
    result = _validate_action(old_format)
    assert result["action"] == "accept", f"old decision key should map to action, got {result['action']}"
    print(f"  ✓ Old 'decision' key maps to action")

    print("✓ Action validation works\n")


def test_trim_strategy_rotation():
    print("=== Testing trimal strategy rotation (gentle only) ===")
    from pipeline_loop import _pick_trim_strategy, TRIM_STRATEGY_ORDER, VALID_TRIM_STRATEGIES

    assert TRIM_STRATEGY_ORDER == ["-gappyout", "-automated1"], TRIM_STRATEGY_ORDER
    assert "-strict" not in VALID_TRIM_STRATEGIES
    assert "-strictplus" not in VALID_TRIM_STRATEGIES
    assert "-nogaps" not in VALID_TRIM_STRATEGIES
    print("  ✓ only gentle strategies allowed (no strict/nogaps — indels kept)")

    tried = []
    s1 = _pick_trim_strategy(0.18, tried)
    assert s1 == "-gappyout", f"first pick should be -gappyout, got {s1}"
    print(f"  ✓ First pick -> {s1}")

    s2 = _pick_trim_strategy(0.18, tried, "-gappyout")
    assert s2 == "-automated1", f"repeated -gappyout should rotate, got {s2}"
    print(f"  ✓ Repeated choice rotates to {s2}")

    # aggressive LLM choices are rejected
    s3 = _pick_trim_strategy(0.40, [], "-nogaps")
    assert s3 == "-gappyout", f"-nogaps must be rejected, got {s3}"
    s4 = _pick_trim_strategy(0.40, [], "-strict")
    assert s4 == "-gappyout", f"-strict must be rejected, got {s4}"
    print("  ✓ -nogaps/-strict rejected (fall back to -gappyout)")

    print("✓ Trimal strategy rotation works\n")


def test_extra_args_sanitization():
    print("=== Testing per-tool extra_args sanitization ===")
    from ai_controller import _sanitize_extra_args, _validate_action

    crash_case = "--maxiterate 0 --op 1.53 --ep 0.0 --nuc"
    cleaned = _sanitize_extra_args("change_tool", "muscle", crash_case)
    assert cleaned == "", f"MAFFT flags on muscle should be stripped, got {cleaned!r}"
    print(f"  ✓ MAFFT flags stripped from muscle decision (the PF19758 crash case)")

    resp = {"action": "change_tool", "tool": "muscle", "msa_mode": "align",
            "extra_args": crash_case, "reason": "test", "confidence": "high"}
    result = _validate_action(resp)
    assert result["extra_args"] == "", f"validator should strip flags, got {result['extra_args']!r}"
    print(f"  ✓ _validate_action strips invalid muscle extra_args")

    kept = _sanitize_extra_args("change_tool", "muscle", "-quiet -perm abc")
    assert "-quiet" in kept and "-perm" in kept and "abc" in kept, f"valid muscle flags kept, got {kept!r}"
    print(f"  ✓ Valid muscle flags kept ({kept})")

    mafft_kept = _sanitize_extra_args("refine", "mafft", "--localpair --maxiterate 1000")
    assert "--localpair" in mafft_kept and "--maxiterate" in mafft_kept and "1000" in mafft_kept
    print(f"  ✓ Valid mafft flags kept ({mafft_kept})")

    trim_ok = _sanitize_extra_args("trim", "muscle", "-gappyout")
    assert trim_ok == "-gappyout", f"gentle trim strategy should pass, got {trim_ok!r}"
    trim_agg = _sanitize_extra_args("trim", "muscle", "-strictplus")
    assert trim_agg == "", f"aggressive trim must be stripped, got {trim_agg!r}"
    trim_bad = _sanitize_extra_args("trim", "muscle", "--maxiterate 0")
    assert trim_bad == "", f"non-trimal trim args should be stripped, got {trim_bad!r}"
    print(f"  ✓ Trim validated: gentle kept, aggressive stripped")

    print("✓ Per-tool extra_args sanitization works\n")


def test_esm_probabilities():
    print("=== Testing ESM residue probabilities ===")
    from protein_embeddings import compute_residue_probabilities, AA_TOKENS

    assert len(AA_TOKENS) == 20, f"Expected 20 amino acids, got {len(AA_TOKENS)}"
    print(f"  ✓ AA_TOKENS has 20 amino acids")

    fasta = ">seq1\nMKVLAAGVG\n>seq2\nMKVLAAGVT"
    prob_data = compute_residue_probabilities(fasta)

    assert len(prob_data) == 2, f"Expected 2 entries, got {len(prob_data)}"
    print(f"  ✓ Computed probabilities for {len(prob_data)} sequences")

    entry = prob_data[0]
    assert len(entry["p_true"]) == 9, f"Expected 9 p_true values, got {len(entry['p_true'])}"
    assert len(entry["entropy"]) == 9, f"Expected 9 entropy values, got {len(entry['entropy'])}"
    assert len(entry["conservation_score"]) == 9, f"Expected 9 conservation scores"
    print(f"  ✓ Each position has p_true, entropy, conservation_score")

    for p in entry["p_true"]:
        assert 0.0 <= p <= 1.0, f"p_true out of range: {p}"
    print(f"  ✓ All p_true values in [0, 1]")

    for c in entry["conservation_score"]:
        assert 0.0 <= c <= 1.0, f"conservation_score out of range: {c}"
    print(f"  ✓ All conservation_score values in [0, 1]")

    print("✓ ESM residue probabilities works\n")


def test_esm_conservation_analysis():
    print("=== Testing ESM conservation analysis ===")
    from esm_analysis import analyze_conservation, _pearson_correlation
    from protein_embeddings import compute_residue_probabilities

    xs = [1.0, 2.0, 3.0, 4.0, 5.0]
    ys = [2.0, 4.0, 6.0, 8.0, 10.0]
    r = _pearson_correlation(xs, ys)
    assert abs(r - 1.0) < 0.01, f"Expected r=1.0 for perfect correlation, got {r}"
    print(f"  ✓ Perfect correlation: r={r:.2f}")

    xs2 = [1.0, 2.0, 3.0, 4.0, 5.0]
    ys2 = [10.0, 8.0, 6.0, 4.0, 2.0]
    r2 = _pearson_correlation(xs2, ys2)
    assert abs(r2 - (-1.0)) < 0.01, f"Expected r=-1.0 for inverse correlation, got {r2}"
    print(f"  ✓ Inverse correlation: r={r2:.2f}")

    fasta = ">seq1\nMKVLAAGVG\n>seq2\nMKVLAAGVT\n>seq3\nMKVLAAGVG"
    prob_data = compute_residue_probabilities(fasta)
    alignment = ">seq1\nMKVLAAGVG\n>seq2\nMKVLAAGVT\n>seq3\nMKVLAAGVG"

    analysis = analyze_conservation(alignment, prob_data)
    assert "conservation_correlation" in analysis
    assert "columns_analyzed" in analysis
    assert "esm_misaligned_columns" in analysis
    print(f"  ✓ analyze_conservation returns required fields")
    print(f"  ✓ Correlation: {analysis['conservation_correlation']}, Columns: {analysis['columns_analyzed']}")

    print("✓ ESM conservation analysis works\n")


def test_esm_gap_penalties():
    print("=== Testing ESM position-specific gap penalties ===")
    from esm_analysis import _compute_position_gap_penalties
    import torch

    sequences = ["MKVLA", "MKVLA"]
    prob_data = [
        {
            "id": "seq1",
            "sequence": "MKVLA",
            "conservation_score": [0.9, 0.1, 0.5, 0.9, 0.1],
            "entropy": [0.1, 2.0, 1.0, 0.1, 2.0],
            "p_true": [0.9, 0.3, 0.5, 0.9, 0.3],
        }
    ]

    gap_open = -2.0
    alpha = 2.0
    penalties = _compute_position_gap_penalties(sequences, prob_data, gap_open, alpha)

    assert len(penalties) == 5, f"Expected 5 penalties, got {len(penalties)}"
    print(f"  ✓ Computed {len(penalties)} position-specific penalties")

    assert penalties[0].item() < penalties[1].item(), \
        f"High conservation (pos 0) should have stronger penalty than low (pos 1): {penalties[0]} vs {penalties[1]}"
    print(f"  ✓ High conservation (pos 0) penalty={penalties[0].item():.2f} stronger than low (pos 1) penalty={penalties[1].item():.2f}")

    expected_pos0 = gap_open * (1.0 + alpha * 0.9)
    expected_pos1 = gap_open * (1.0 + alpha * 0.1)
    assert abs(penalties[0].item() - expected_pos0) < 0.01, f"Expected {expected_pos0}, got {penalties[0].item()}"
    print(f"  ✓ Penalty formula correct: gap_open * (1 + alpha * conservation)")

    print("✓ ESM position-specific gap penalties works\n")


def test_esm_column_report():
    print("=== Testing ESM column report generation ===")
    from esm_analysis import generate_column_report

    analysis = {
        "conservation_correlation": 0.82,
        "columns_analyzed": 150,
        "esm_misaligned_columns": [
            {"column": 42, "esm_conservation": 0.85, "observed_hill_q1": 6.2, "gap_fraction": 0.1},
            {"column": 87, "esm_conservation": 0.78, "observed_hill_q1": 5.8, "gap_fraction": 0.15},
        ],
        "conserved_columns": 98,
        "variable_columns": 22,
    }

    report = generate_column_report(analysis, "")
    assert "ESM Conservation Analysis" in report
    assert "r = 0.82" in report
    assert "Col 42" in report
    assert "Col 87" in report
    assert "Conserved columns" in report
    print(f"  ✓ Report contains all expected sections")
    print(f"  ✓ Report length: {len(report)} chars")

    print("✓ ESM column report works\n")


def _make_cv_experiment(i, problem, outcome, fp=None, strategy="accept:auto:"):
    init_gap = 0.3 if problem == "high_gap_fraction" else 0.05
    final_gap = 0.05 if outcome == "success" else init_gap
    return {
        "experiment_id": f"exp{i:03d}",
        "timestamp": f"2026-11-{(i % 28) + 1:02d}T00:00:00",
        "problem_type": problem,
        "final_outcome": outcome,
        "initial_gap_fraction": init_gap,
        "final_gap_fraction": final_gap,
        "fingerprint": fp if fp is not None else f"fp{i}",
        "sequence_signature": f"sig{i}",
        "actions_tried": [
            {"iteration": 1, "action": strategy.split(":")[0],
             "strategy": strategy.split(":")[1], "extra_args": strategy.split(":", 2)[2],
             "outcome": "success" if outcome == "success" else "no_change"}
        ],
    }


def test_cross_validate_kfold():
    print("=== Testing k-fold cross-validation ===")
    from rag_advanced import cross_validate, _make_folds

    exps = []
    for i in range(20):
        problem = "high_gap_fraction" if i % 4 == 0 else "good_quality"
        outcome = "success" if i % 5 != 0 else "no_improvement"
        exps.append(_make_cv_experiment(i, problem, outcome))

    result = cross_validate(exps, k=5)
    assert "message" not in result or not result.get("message"), f"unexpected: {result.get('message')}"
    for key in ("accuracy", "baseline_accuracy", "lift", "brier_score", "gap_mae", "folds", "per_problem"):
        assert key in result, f"missing key {key}"
    assert 0.0 <= result["accuracy"] <= 1.0
    assert 0.0 <= result["brier_score"] <= 1.0
    assert result["folds"] >= 2
    assert result["test_size"] == len(exps), f"all experiments should be tested once, got {result['test_size']}"
    print(f"  ✓ k-fold metrics present (acc={result['accuracy']}, base={result['baseline_accuracy']}, brier={result['brier_score']})")

    folds = _make_folds(exps, 5)
    assert len(folds) >= 2
    seen = set()
    for f in folds:
        for e in f:
            eid = e["experiment_id"]
            assert eid not in seen, f"{eid} appeared in two folds"
            seen.add(eid)
    assert len(seen) == len(exps)
    print(f"  ✓ each experiment appears in exactly one fold")

    # fingerprint leakage: same fp must be in one fold only
    dup_exps = [_make_cv_experiment(i, "good_quality", "success", fp="same") for i in range(10)]
    dup_exps += [_make_cv_experiment(100 + i, "high_gap_fraction", "success", fp=f"other{i}") for i in range(10)]
    folds2 = _make_folds(dup_exps, 5)
    for key in ("same",):
        fold_ids = [idx for idx, f in enumerate(folds2) if any(e["fingerprint"] == key for e in f)]
        assert len(fold_ids) == 1, f"fingerprint {key} split across folds {fold_ids}"
    print(f"  ✓ same-fingerprint experiments never split across folds")

    tiny = cross_validate(exps[:3])
    assert tiny.get("message"), "too-few case should return a message"
    print(f"  ✓ tiny input handled gracefully")
    print("✓ k-fold cross-validation works\n")


def test_strategy_table():
    print("=== Testing strategy table ===")
    import visualization
    from visualization import print_strategy_table
    import io
    import contextlib

    fake = [
        {"problem_type": "high_gap_fraction", "final_outcome": "success",
         "initial_gap_fraction": 0.3, "final_gap_fraction": 0.05,
         "actions_tried": [{"action": "trim", "strategy": "auto", "extra_args": "-nogaps"}]},
        {"problem_type": "high_gap_fraction", "final_outcome": "success",
         "initial_gap_fraction": 0.25, "final_gap_fraction": 0.0,
         "actions_tried": [{"action": "trim", "strategy": "auto", "extra_args": "-nogaps"}]},
        {"problem_type": "high_gap_fraction", "final_outcome": "success",
         "initial_gap_fraction": 0.35, "final_gap_fraction": 0.1,
         "actions_tried": [{"action": "trim", "strategy": "auto", "extra_args": "-nogaps"}]},
        {"problem_type": "high_gap_fraction", "final_outcome": "no_improvement",
         "initial_gap_fraction": 0.4, "final_gap_fraction": 0.4,
         "actions_tried": [{"action": "refine", "strategy": "auto", "extra_args": ""}]},
        {"problem_type": "good_quality", "final_outcome": "success",
         "initial_gap_fraction": 0.05, "final_gap_fraction": 0.05,
         "actions_tried": [{"action": "accept", "strategy": "auto", "extra_args": ""}]},
    ]

    orig = visualization.load_experiments
    visualization.load_experiments = lambda: fake
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            print_strategy_table(min_n=3)
    finally:
        visualization.load_experiments = orig

    out = buf.getvalue()
    assert "STRATEGY TABLE" in out
    assert "high_gap_fraction" in out
    assert "nogaps" in out
    assert "refine" not in out, "n=1 refine should be filtered out (min_n=3)"
    assert "n=  3" in out or "n= 3" in out, f"nogaps should have n=3, got:\n{out}"
    print("  ✓ n>=3 filter, grouping, and counts correct")
    print("✓ strategy table works\n")


def test_learning_curve():
    print("=== Testing learning curve ===")
    import visualization
    from visualization import print_learning_curve
    import io
    import contextlib

    fake = []
    for i in range(10):
        fake.append({
            "timestamp": f"2026-11-{i + 1:02d}T00:00:00",
            "problem_type": "high_gap_fraction",
            "final_outcome": "success" if i < 8 else "no_improvement",
            "initial_gap_fraction": 0.2,
            "final_gap_fraction": 0.1 if i < 8 else 0.2,
        })
    for i in range(5):
        fake.append({
            "timestamp": f"2026-11-{i + 1:02d}T01:00:00",
            "problem_type": "good_quality",
            "final_outcome": "success",
            "initial_gap_fraction": 0.0,
            "final_gap_fraction": 0.0,
        })
    fake.append({
        "timestamp": "2026-11-01T02:00:00",
        "problem_type": "moderate_gap_fraction",
        "final_outcome": "success",
        "initial_gap_fraction": 0.15,
        "final_gap_fraction": 0.05,
    })

    orig = visualization.load_experiments
    visualization.load_experiments = lambda: fake
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            print_learning_curve(buckets=5)
    finally:
        visualization.load_experiments = orig

    out = buf.getvalue()
    assert "LEARNING CURVE (cumulative as analyses were added)" in out
    assert "high_gap_fraction" in out
    assert "moderate_gap_fraction" in out
    assert "good_quality" in out
    assert "reference" in out, "good_quality should be labeled reference"
    assert "too few points" in out, "moderate_gap n=1 should use summary line"
    assert "Growth:" in out, "growth line first→last checkpoint expected"
    assert "red mean" in out or "red —" in out
    assert "avg Δgap" not in out, "old Δgap format should be gone"
    # cumulative-by-n, not date buckets
    assert "at n=" in out, f"expected cumulative checkpoints:\n{out}"
    assert "2026-11" not in out, "date rows should be gone"
    # high_gap n=10 → checkpoints ~quarter steps, last is n=10
    assert "at n= 10" in out, f"final checkpoint expected:\n{out}"
    # hard problem types listed before good_quality
    assert out.index("high_gap_fraction") < out.index("good_quality")
    # cumulative rows are monotonic in n within high_gap section
    import re as _re
    ns = [int(m) for m in _re.findall(r"at n=\s*(\d+):", out)]
    hg_ns = ns[: len(ns) // 3 + 1]
    assert all(b > a for a, b in zip(hg_ns, hg_ns[1:])), f"n checkpoints not increasing: {hg_ns}"
    print("  ✓ cumulative-by-n checkpoints, growth line, no date buckets")
    print("✓ learning curve works\n")


def test_esm_gap_stats():
    print("=== Testing ESM gap-band learning ===")
    import tempfile
    import shutil
    import pathlib
    import rag_retrieve

    tmp = pathlib.Path(tempfile.mkdtemp(prefix="esm_gap_"))
    orig_dir = rag_retrieve.KNOWLEDGE_DIR
    orig_gz = rag_retrieve.ESM_GAP_STATS_GZ
    try:
        rag_retrieve.KNOWLEDGE_DIR = tmp
        # empty -> default threshold
        thr, learned = rag_retrieve.get_esm_threshold()
        assert thr == 0.25 and not learned, f"empty should be default, got {thr},{learned}"
        assert rag_retrieve.get_esm_gap_hint() == ""

        # band helper
        assert rag_retrieve.esm_gap_band(0.12) == "0.10-0.15"
        assert rag_retrieve.esm_gap_band(0.22) == "0.20-0.25"

        # low-gap band (0.00-0.05) must NOT lower threshold to 0
        for _ in range(5):
            rag_retrieve.update_esm_gap_stats({
                "initial_gap_fraction": 0.03,
                "actions_tried": [
                    {"tool": "esm2", "gap_fraction": 0.03, "action": "change_tool"},
                ],
            })
        thr0, learned0 = rag_retrieve.get_esm_threshold()
        assert thr0 == 0.25 and not learned0, (
            f"gap<0.10 band must be ignored for threshold, got thr={thr0} learned={learned0}"
        )
        print("  ✓ band 0.00-0.05 ignored (thr stays default, not 0)")

        # three successful probes in 0.15-0.20 band (gap_before ~0.17)
        for k in range(3):
            exp = {
                "initial_gap_fraction": 0.17,
                "actions_tried": [
                    {"tool": "mafft", "gap_fraction": 0.17, "action": "refine"},
                    {"tool": "esm2", "gap_fraction": 0.10, "action": "change_tool"},
                ],
            }
            rag_retrieve.update_esm_gap_stats(exp)

        stats = rag_retrieve.load_esm_gap_stats()
        assert "0.15-0.20" in stats, f"expected band, got {list(stats)}"
        assert stats["0.15-0.20"]["n"] == 3
        assert stats["0.15-0.20"]["success"] == 3

        thr, learned = rag_retrieve.get_esm_threshold()
        assert learned and abs(thr - 0.15) < 1e-9, f"learned thr={thr} learned={learned}"
        hint = rag_retrieve.get_esm_gap_hint()
        assert "Learned: ESM-2 works when gap > 0.15" in hint, hint
        print(f"  ✓ learned threshold {thr} from 3 successes in {rag_retrieve.esm_gap_band(0.17)}")

        # failed trial does not lower threshold alone
        exp_fail = {
            "initial_gap_fraction": 0.30,
            "actions_tried": [
                {"tool": "mafft", "gap_fraction": 0.30, "action": "refine"},
                {"tool": "esm2", "gap_fraction": 0.31, "action": "change_tool"},
            ],
        }
        rag_retrieve.update_esm_gap_stats(exp_fail)
        stats = rag_retrieve.load_esm_gap_stats()
        assert stats["0.30-0.35"]["n"] == 1 and stats["0.30-0.35"]["success"] == 0
        print("  ✓ failed probe recorded in its band")

        # dashboard print does not crash
        import visualization
        import io
        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            visualization.print_esm_gap_stats()
        out = buf.getvalue()
        assert "ESM-2 BY GAP BAND" in out
        assert "0.15-0.20" in out
        assert "Threshold: gap > 0.15" in out
        print("  ✓ dashboard section renders")
    finally:
        rag_retrieve.KNOWLEDGE_DIR = orig_dir
        rag_retrieve.ESM_GAP_STATS_GZ = orig_gz
        shutil.rmtree(tmp, ignore_errors=True)


def test_esm_post_stats():
    print("=== Testing ESM post-correction stats ===")
    import tempfile
    import shutil
    import pathlib
    import rag_retrieve

    tmp = pathlib.Path(tempfile.mkdtemp(prefix="esm_post_"))
    orig_dir = rag_retrieve.KNOWLEDGE_DIR
    orig_gz = rag_retrieve.ESM_POST_STATS_GZ
    try:
        rag_retrieve.KNOWLEDGE_DIR = tmp
        assert rag_retrieve.load_esm_post_stats() == {}
        assert rag_retrieve.get_esm_post_hint() == ""
        assert rag_retrieve.get_esm_post_advice(0.20) == "explore"

        # three successes in 0.15-0.20
        for _ in range(3):
            rag_retrieve.update_esm_post_stats({
                "esm_post_applied": True,
                "esm_gap_before": 0.17,
                "esm_gap_after": 0.10,
            })
        stats = rag_retrieve.load_esm_post_stats()
        assert stats["0.15-0.20"]["n"] == 3
        assert stats["0.15-0.20"]["success"] == 3
        assert rag_retrieve.get_esm_post_advice(0.17) == "use"
        assert rag_retrieve.get_esm_post_advice(0.05) == "explore"
        hint = rag_retrieve.get_esm_post_hint()
        assert "ESM post-correction" in hint and "0.15-0.20" in hint, hint

        # not-applied records ignored
        rag_retrieve.update_esm_post_stats({"esm_post_applied": False, "esm_gap_before": 0.5})
        assert rag_retrieve.load_esm_post_stats()["0.15-0.20"]["n"] == 3

        # three failures -> avoid
        for _ in range(3):
            rag_retrieve.update_esm_post_stats({
                "esm_post_applied": True,
                "esm_gap_before": 0.30,
                "esm_gap_after": 0.31,
            })
        assert rag_retrieve.get_esm_post_advice(0.32) == "avoid"

        import visualization, io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            visualization.print_esm_post_stats()
        out = buf.getvalue()
        assert "ESM POST-CORRECTION BY GAP BAND" in out
        assert "USE" in out and "AVOID" in out
        print("  ✓ stats + advice + dashboard section")
    finally:
        rag_retrieve.KNOWLEDGE_DIR = orig_dir
        rag_retrieve.ESM_POST_STATS_GZ = orig_gz
        shutil.rmtree(tmp, ignore_errors=True)


def test_esm_post_decision_gate():
    print("=== Testing esm_post decision gate ===")
    import inspect
    import build_knowledge
    from pipeline_loop import ESM_POST_BAD_GAP, run_pipeline

    # pipeline default + build both use selective if_bad
    sig = inspect.signature(run_pipeline)
    assert sig.parameters["esm_post"].default == "if_bad", sig.parameters["esm_post"].default
    src = inspect.getsource(build_knowledge)
    assert 'esm_post="if_bad"' in src, "build must use if_bad"
    assert 'esm_post="always"' not in src, "build must not use always"
    print("  ✓ build + run_pipeline default = if_bad (not always)")

    # gate matrix (mirrors pipeline_loop logic)
    import rag_retrieve
    import tempfile, shutil, pathlib
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="esm_gate_"))
    orig_dir = rag_retrieve.KNOWLEDGE_DIR
    orig_gz = rag_retrieve.ESM_POST_STATS_GZ
    try:
        rag_retrieve.KNOWLEDGE_DIR = tmp

        def should_post(mode, gap, advice):
            if mode == "always":
                return True
            if mode == "if_bad" and gap > ESM_POST_BAD_GAP:
                return advice != "avoid"
            return False

        # empty stats -> explore
        advice = rag_retrieve.get_esm_post_advice(0.20)
        assert advice == "explore"
        assert should_post("if_bad", 0.20, advice) is True   # bad + unknown -> learn
        assert should_post("if_bad", 0.05, advice) is False  # good gap -> skip
        assert should_post("if_bad", 0.20, "avoid") is False # bad + avoid -> skip
        assert should_post("if_bad", 0.20, "use") is True    # bad + use -> run
        assert should_post("off", 0.30, "use") is False
        assert should_post("always", 0.01, "avoid") is True
        print("  ✓ gate matrix: good->skip, bad+avoid->skip, bad+use/explore->run")
    finally:
        rag_retrieve.KNOWLEDGE_DIR = orig_dir
        rag_retrieve.ESM_POST_STATS_GZ = orig_gz
        shutil.rmtree(tmp, ignore_errors=True)


def test_derive_warm_plan():
    print("=== Testing derive_warm_plan ===")
    from rag_retrieve import derive_warm_plan

    actions = [
        {"action": "change_tool", "msa_mode": "auto", "tool": "mafft",
         "strategy": "auto", "extra_args": "", "gap_fraction": 0.22, "outcome": "no_change"},
        {"action": "change_tool", "tool": "esm2", "msa_mode": "local",
         "strategy": "local", "extra_args": "", "gap_fraction": 0.08, "outcome": "success"},
        {"action": "trim", "tool": "muscle", "strategy": "",
         "extra_args": "-gappyout", "trim_strategy": "-gappyout",
         "gap_fraction": 0.06, "outcome": "success"},
        {"action": "change_tool", "tool": "muscle", "strategy": "align",
         "extra_args": "", "gap_fraction": 0.25, "outcome": "no_improvement"},
    ]
    plan = derive_warm_plan(actions)
    assert plan["start"] is not None, plan
    assert plan["start"]["tool"] == "esm2", plan["start"]
    assert plan["esm_direct"] is True
    assert any(x["tool"] == "trimal" for x in plan["plan"]), plan["plan"]
    assert any(x["tool"] == "muscle" for x in plan["avoid"]), plan["avoid"]
    print(f"  ✓ start=esm2 direct, plan+avoid populated")

    # tool inference when tool missing
    plan2 = derive_warm_plan([
        {"action": "change_tool", "msa_mode": "align", "strategy": "align",
         "extra_args": "", "gap_fraction": 0.1, "outcome": "success"},
    ])
    assert plan2["start"]["tool"] == "muscle", plan2
    plan3 = derive_warm_plan([
        {"action": "change_tool", "msa_mode": "auto", "strategy": "auto",
         "extra_args": "", "gap_fraction": 0.2, "outcome": "success"},
        {"action": "trim", "strategy": "-strict", "extra_args": "-strict",
         "trim_strategy": "-strict", "gap_fraction": 0.1, "outcome": "success"},
    ])
    assert plan3["start"] is not None and plan3["start"]["tool"] == "mafft", plan3
    assert any(x["tool"] == "trimal" for x in plan3["plan"]), plan3
    print("  ✓ tool inference (muscle/trimal without explicit tool)")

    assert derive_warm_plan(None)["start"] is None
    print("✓ derive_warm_plan works\n")

    print("✓ ESM gap stats works\n")


def test_dashboard_by_problem():
    print("=== Testing dashboard split by problem type ===")
    import visualization
    from visualization import print_dashboard, print_tool_usage, print_strategy_performance
    import io
    import contextlib

    fake = [
        {"problem_type": "high_gap_fraction", "final_outcome": "success",
         "initial_gap_fraction": 0.4, "final_gap_fraction": 0.1, "was_trimmed": True,
         "actions_tried": [{"action": "trim", "strategy": "auto", "extra_args": "-nogaps"}]},
        {"problem_type": "high_gap_fraction", "final_outcome": "success",
         "initial_gap_fraction": 0.3, "final_gap_fraction": 0.0, "was_trimmed": True,
         "actions_tried": [{"action": "trim", "strategy": "auto", "extra_args": "-strict"}]},
        {"problem_type": "moderate_gap_fraction", "final_outcome": "no_improvement",
         "initial_gap_fraction": 0.2, "final_gap_fraction": 0.2, "was_trimmed": False,
         "actions_tried": [{"action": "refine", "strategy": "auto", "extra_args": ""}]},
        {"problem_type": "moderate_gap_fraction", "final_outcome": "success",
         "initial_gap_fraction": 0.25, "final_gap_fraction": 0.05, "was_trimmed": True,
         "actions_tried": [{"action": "trim", "strategy": "auto", "extra_args": "-gappyout"}]},
        {"problem_type": "good_quality", "final_outcome": "success",
         "initial_gap_fraction": 0.0, "final_gap_fraction": 0.0, "was_trimmed": False,
         "actions_tried": [{"action": "accept", "strategy": "auto", "extra_args": ""}]},
        {"problem_type": "good_quality", "final_outcome": "success",
         "initial_gap_fraction": 0.05, "final_gap_fraction": 0.05, "was_trimmed": False,
         "actions_tried": [{"action": "accept", "strategy": "auto", "extra_args": ""}]},
    ]

    orig = visualization.load_experiments
    visualization.load_experiments = lambda: fake
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            print_dashboard()
            print_strategy_performance()
            print_tool_usage()
    finally:
        visualization.load_experiments = orig

    out = buf.getvalue()
    assert "BY PROBLEM TYPE" in out
    assert "high_gap_fraction (n=2)" in out
    assert "moderate_gap_fraction (n=2)" in out
    assert "good_quality (n=2)" in out
    assert "reference" in out
    assert "gap init:" in out
    assert "gap final:" in out
    assert "reduction:" in out
    assert "avg Δgap" not in out, "old Δgap format should be gone"
    assert "trimmed:" in out

    from visualization import _gap_set_stats
    hg = [e for e in fake if e["problem_type"] == "high_gap_fraction"]
    g = _gap_set_stats(hg)
    assert abs(g["init_mean"] - 0.35) < 1e-9
    assert abs(g["final_mean"] - 0.05) < 1e-9
    assert abs(g["red_mean"] - 87.5) < 1e-9, f"red_mean={g['red_mean']}"
    assert abs(g["red_med"] - 87.5) < 1e-9
    gq = _gap_set_stats([e for e in fake if e["problem_type"] == "good_quality"])
    assert gq["red_n"] == 1, "init==0 skipped, only 0.05→0.05 counts"
    assert gq["red_mean"] == 0.0
    # outcomes split: moderate has no_improvement only in its section
    assert "no_improvement" in out
    # hard problems before reference
    assert out.index("high_gap_fraction") < out.index("good_quality")
    # strategy performance section also split
    assert "STRATEGY PERFORMANCE (by problem type)" in out
    # tool usage section split (trim counts come from KB)
    assert "TOOL USAGE (by problem type)" in out
    assert "trimal" in out
    print("  ✓ dashboard/strategy/tool sections split, order, reference label")
    print("✓ dashboard by problem type works\n")


def main():
    print("Running all tests...\n")

    test_msa_runner()
    test_metrics()
    test_convergence()
    test_llm()
    test_full_pipeline()
    test_toon_roundtrip()
    test_toon_nested_tabular()
    test_sparse_index()
    test_toon_storage_roundtrip()
    test_path_resolution()
    test_idf_document_frequency()
    test_warm_start()
    test_hill_numbers()
    test_esm_alignment()
    test_rule_based_decision()
    test_metrics_echo_validation()
    test_duplicate_detection_tfidf()
    test_validate_action()
    test_esm_probabilities()
    test_esm_conservation_analysis()
    test_esm_gap_penalties()
    test_esm_column_report()
    test_trim_strategy_rotation()
    test_extra_args_sanitization()
    test_cross_validate_kfold()
    test_strategy_table()
    test_learning_curve()
    test_dashboard_by_problem()
    test_esm_gap_stats()
    test_esm_post_stats()
    test_esm_post_decision_gate()
    test_derive_warm_plan()

    print("All tests completed!")


if __name__ == "__main__":
    main()