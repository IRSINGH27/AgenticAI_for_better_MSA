"""
Storage format tools: inspect, migrate, and compare knowledge base formats.

Supports the three-layer storage:
  - TOON files (text, LLM-friendly)
  - JSON config (hand-editable)
  - Sparse NumPy index (fast search)
"""
import sys
import json
import pathlib

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
KNOWLEDGE_DIR = SCRIPT_DIR.parent / "knowledge"

TOON_FILE = "experiments.toon"
JSONL_FILE = "experiments.jsonl"
STRATEGY_STATS_TOON = "strategy_stats.toon"
STRATEGY_STATS_JSON = "strategy_stats.json"
INDEX_PREFIX = "tfidf_index"


def _human_size(num_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if num_bytes < 1024:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} TB"


def cmd_format_info() -> None:
    print("=" * 60)
    print("STORAGE FORMAT INFORMATION")
    print("=" * 60)
    print()

    toon_path = KNOWLEDGE_DIR / TOON_FILE
    jsonl_path = KNOWLEDGE_DIR / JSONL_FILE
    stats_toon = KNOWLEDGE_DIR / STRATEGY_STATS_TOON
    stats_json = KNOWLEDGE_DIR / STRATEGY_STATS_JSON
    npz_path = KNOWLEDGE_DIR / (INDEX_PREFIX + ".npz")
    vocab_path = KNOWLEDGE_DIR / (INDEX_PREFIX + "_vocab.json")

    print("Layer 1: TOON (text, LLM-friendly)")
    if toon_path.exists():
        size = toon_path.stat().st_size
        lines = len(toon_path.read_text().splitlines())
        print(f"  [OK] {TOON_FILE:<25} {_human_size(size):>10}   {lines} lines")
    else:
        print(f"  [MISSING] {TOON_FILE}")

    if stats_toon.exists():
        size = stats_toon.stat().st_size
        lines = len(stats_toon.read_text().splitlines())
        print(f"  [OK] {STRATEGY_STATS_TOON:<25} {_human_size(size):>10}   {lines} lines")
    else:
        print(f"  [MISSING] {STRATEGY_STATS_TOON}")

    print()
    print("Layer 2: JSON (legacy / config)")
    if jsonl_path.exists():
        size = jsonl_path.stat().st_size
        n = sum(1 for line in jsonl_path.read_text().splitlines() if line.strip())
        print(f"  [OK] {JSONL_FILE:<25} {_human_size(size):>10}   {n} records (legacy)")
    else:
        print(f"  [NONE]  {JSONL_FILE} (no legacy data)")

    if stats_json.exists():
        size = stats_json.stat().st_size
        print(f"  [OK] {STRATEGY_STATS_JSON:<25} {_human_size(size):>10}   (legacy)")
    else:
        print(f"  [NONE]  {STRATEGY_STATS_JSON} (no legacy data)")

    print()
    print("Layer 3: Sparse NumPy (fast search)")
    if npz_path.exists() and vocab_path.exists():
        npz_size = npz_path.stat().st_size
        vocab_size = vocab_path.stat().st_size
        print(f"  [OK] {INDEX_PREFIX + '.npz':<25} {_human_size(npz_size):>10}   (sparse matrix)")
        print(f"  [OK] {INDEX_PREFIX + '_vocab.json':<25} {_human_size(vocab_size):>10}   (vocab + ids)")

        try:
            from tfidf_sparse import load_index, get_index_stats
            matrix, vocab, ids = load_index(KNOWLEDGE_DIR)
            stats = get_index_stats(matrix, vocab)
            print()
            print("  Sparse index contents:")
            for k, v in stats.items():
                print(f"    {k}: {v}")
        except Exception as e:
            print(f"  [WARN] Could not read sparse index: {e}")
    else:
        print(f"  [MISSING] {INDEX_PREFIX}.npz or {INDEX_PREFIX}_vocab.json")

    print()
    print("Format priority: TOON + Sparse (active), JSONL (fallback reader only)")


def cmd_migrate_to_toon() -> None:
    print("=" * 60)
    print("MIGRATE: JSONL -> TOON + Sparse Index")
    print("=" * 60)
    print()

    jsonl_path = KNOWLEDGE_DIR / JSONL_FILE
    stats_json = KNOWLEDGE_DIR / STRATEGY_STATS_JSON
    toon_path = KNOWLEDGE_DIR / TOON_FILE

    if not jsonl_path.exists():
        print(f"No {JSONL_FILE} to migrate.")
        return

    if toon_path.exists():
        toon_size = toon_path.stat().st_size
        print(f"WARNING: {TOON_FILE} already exists ({_human_size(toon_size)}).")
        resp = input("Overwrite? [y/N]: ").strip().lower()
        if resp != "y":
            print("Aborted.")
            return

    from toon_format import to_toon, from_toon
    from tfidf_sparse import build_sparse_index, save_index

    records = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    print(f"Read {len(records)} records from {JSONL_FILE}")

    toon_path.write_text(to_toon(records))
    print(f"  Wrote {TOON_FILE} ({_human_size(toon_path.stat().st_size)})")

    if stats_json.exists():
        try:
            stats = json.load(open(stats_json))
            stats_toon = KNOWLEDGE_DIR / STRATEGY_STATS_TOON
            stats_toon.write_text(to_toon({"strategy_stats": stats}))
            print(f"  Wrote {STRATEGY_STATS_TOON}")
        except Exception as e:
            print(f"  [WARN] Failed to migrate strategy_stats: {e}")

    if records:
        matrix, vocab, ids = build_sparse_index(records)
        save_index(matrix, vocab, ids, KNOWLEDGE_DIR)
        print(f"  Wrote {INDEX_PREFIX}.npz + {INDEX_PREFIX}_vocab.json")
        print(f"    Matrix: {matrix.shape}, nnz={matrix.nnz}, vocab={len(vocab)}")

    print()
    print("Migration complete. Old JSONL files preserved (not deleted).")
    print("Run --format-info to verify.")


def cmd_sparse_info() -> None:
    print("=" * 60)
    print("SPARSE INDEX INFORMATION")
    print("=" * 60)
    print()

    from tfidf_sparse import load_index, get_index_stats

    matrix, vocab, ids = load_index(KNOWLEDGE_DIR)
    if matrix.shape[0] == 0:
        print("Sparse index is empty.")
        return

    stats = get_index_stats(matrix, vocab)
    print("Matrix shape:", stats["matrix_shape"])
    print("Documents:   ", stats["num_documents"])
    print("Vocab size:  ", stats["vocab_size"])
    print("Non-zeros:   ", stats["nnz"])
    print("Density:     ", f"{stats['density']*100:.4f}%")
    print("Avg terms/doc:", f"{stats['avg_terms_per_doc']:.2f}")
    print()
    print("Experiment IDs in index:")
    for i, eid in enumerate(ids):
        print(f"  [{i}] {eid}")
    print()
    print("Top 20 vocab terms (alphabetical):")
    for term in sorted(vocab.keys())[:20]:
        print(f"  {term}: id={vocab[term]}")


def cmd_toon_context() -> None:
    print("=" * 60)
    print("TOON CONTEXT PREVIEW (what the LLM sees)")
    print("=" * 60)
    print()

    from rag_retrieve import load_knowledge, format_retrieved

    records = load_knowledge()
    if not records:
        print("No experiments in knowledge base.")
        return

    problem_types = set(r.get("problem_type", "general") for r in records)
    print(f"Found {len(records)} experiments across problem types: {problem_types}")
    print()

    for pt in sorted(problem_types):
        relevant = [r for r in records if r.get("problem_type") == pt][:3]
        print("-" * 60)
        print(f"PROBLEM TYPE: {pt} (showing {len(relevant)} records)")
        print("-" * 60)
        context = format_retrieved(relevant, problem_type=pt)
        print(context)
        print()


def cmd_compare_formats() -> None:
    print("=" * 60)
    print("FORMAT COMPARISON")
    print("=" * 60)
    print()

    toon_path = KNOWLEDGE_DIR / TOON_FILE
    jsonl_path = KNOWLEDGE_DIR / JSONL_FILE
    npz_path = KNOWLEDGE_DIR / (INDEX_PREFIX + ".npz")
    vocab_path = KNOWLEDGE_DIR / (INDEX_PREFIX + "_vocab.json")

    print(f"{'Format':<20} {'Size':>12}   {'Records':>10}   Status")
    print("-" * 60)

    if toon_path.exists():
        size = toon_path.stat().st_size
        text = toon_path.read_text()
        try:
            from toon_format import from_toon
            data = from_toon(text)
            n = len(data) if isinstance(data, list) else sum(1 for v in data.values() if isinstance(v, list))
        except Exception:
            n = "?"
        print(f"{'TOON':<20} {_human_size(size):>12}   {str(n):>10}   [active]")
    else:
        print(f"{'TOON':<20} {'-':>12}   {'-':>10}   [missing]")

    if jsonl_path.exists():
        size = jsonl_path.stat().st_size
        n = sum(1 for line in jsonl_path.read_text().splitlines() if line.strip())
        print(f"{'JSONL':<20} {_human_size(size):>12}   {str(n):>10}   [legacy]")
    else:
        print(f"{'JSONL':<20} {'-':>12}   {'-':>10}   [none]")

    if npz_path.exists():
        size = npz_path.stat().st_size
        try:
            from tfidf_sparse import load_index
            matrix, vocab, ids = load_index(KNOWLEDGE_DIR)
            n = matrix.shape[0]
        except Exception:
            n = "?"
        print(f"{'Sparse .npz':<20} {_human_size(size):>12}   {str(n):>10}   [active]")
    else:
        print(f"{'Sparse .npz':<20} {'-':>12}   {'-':>10}   [missing]")

    print()
    print("Notes:")
    print("  - TOON is the source of truth for experiment records (LLM-readable)")
    print("  - Sparse matrix accelerates retrieval (cosine sim via single matmul)")
    print("  - JSONL is kept for backward compatibility; readers auto-fallback")


def cmd_inspect(experiment_id: str = None) -> None:
    print("=" * 60)
    if experiment_id:
        print(f"EXPERIMENT INSPECTION: {experiment_id}")
    else:
        print("EXPERIMENT INSPECTION (no ID provided)")
    print("=" * 60)
    print()

    from rag_retrieve import load_knowledge

    records = load_knowledge()
    if not records:
        print("No experiments in knowledge base.")
        return

    if not experiment_id:
        print("Available experiment IDs:")
        for r in records:
            eid = r.get("experiment_id", "?")
            pt = r.get("problem_type", "?")
            gap = r.get("final_gap_fraction", 0)
            outcome = r.get("final_outcome", "?")
            print(f"  {eid:<12} {pt:<20} gap={gap:.3f}  outcome={outcome}")
        print()
        print("Usage: python dashboard.py --inspect <experiment_id>")
        return

    target = None
    for r in records:
        if r.get("experiment_id") == experiment_id:
            target = r
            break

    if not target:
        print(f"Experiment '{experiment_id}' not found.")
        print()
        print("Available experiment IDs:")
        for r in records:
            print(f"  {r.get('experiment_id', '?')}")
        return

    print("Identity:")
    print(f"  experiment_id: {target.get('experiment_id')}")
    print(f"  timestamp:     {target.get('timestamp', '?')}")
    print()

    print("Problem:")
    print(f"  problem_type:  {target.get('problem_type', '?')}")
    print(f"  num_sequences: {target.get('num_sequences', '?')}")
    print(f"  avg_seq_len:   {target.get('avg_sequence_length', '?')}")
    print()

    print("Gap Analysis:")
    print(f"  initial_gap:   {target.get('initial_gap_fraction', 0):.3f}")
    print(f"  final_gap:     {target.get('final_gap_fraction', 0):.3f}")
    print(f"  was_trimmed:   {target.get('was_trimmed', False)}")
    print(f"  final_outcome: {target.get('final_outcome', '?')}")
    print()

    metrics = target.get("extended_metrics", {})
    if metrics:
        print("Extended Metrics:")
        for k, v in metrics.items():
            print(f"  {k:<20} {v}")
        print()

    actions = target.get("actions_tried", [])
    if actions:
        print(f"Actions Tried ({len(actions)} iterations):")
        for a in actions:
            it = a.get("iteration", "?")
            action = a.get("action", "?")
            strategy = a.get("strategy", "?")
            extra = a.get("extra_args", "")
            gap = a.get("gap_fraction", 0)
            outcome = a.get("outcome", "?")
            extra_str = f" --{extra}" if extra else ""
            print(f"  iter {it}: {action} {strategy}{extra_str} -> gap={gap:.3f} ({outcome})")
        print()

    lesson = target.get("lesson", "")
    if lesson:
        print("Lesson Learned:")
        print(f"  {lesson}")
        print()

    sig = target.get("sequence_signature", [])
    if sig:
        print(f"Sequence Signature (k=5): {len(sig)} terms")
        print(f"  {sig[:10]}{'...' if len(sig) > 10 else ''}")
        print()

    embedding = target.get("tfidf_embedding", {})
    if embedding:
        print(f"TF-IDF Embedding (sparse): {len(embedding)} terms")
        top_terms = sorted(embedding.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        for term, val in top_terms:
            print(f"  {term}: {val:.4f}")
        print()

    fingerprint = target.get("fingerprint", {})
    if fingerprint:
        print("Sequence Fingerprint:")
        for k, v in fingerprint.items():
            if k != "char_freq":
                print(f"  {k}: {v}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python format_tools.py [format-info|migrate-to-toon|sparse-info|toon-context|compare-formats|inspect]")
        sys.exit(1)
    cmd = sys.argv[1]
    {
        "format-info": cmd_format_info,
        "migrate-to-toon": cmd_migrate_to_toon,
        "sparse-info": cmd_sparse_info,
        "toon-context": cmd_toon_context,
        "compare-formats": cmd_compare_formats,
        "inspect": cmd_inspect,
    }.get(cmd, lambda: print(f"Unknown command: {cmd}"))()
