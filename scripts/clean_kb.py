import json
import shutil
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.resolve()))

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
KNOWLEDGE_DIR = SCRIPT_DIR.parent / "knowledge"
BACKUP_DIR = KNOWLEDGE_DIR / "backup"
JSONL_FILE = KNOWLEDGE_DIR / "experiments.jsonl"
TOON_FILE = KNOWLEDGE_DIR / "experiments.toon"
MAX_RECORD_BYTES = 1_000_000


def backup_files():
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    for name in ["experiments.jsonl", "experiments.toon", "tfidf_index.npz", "tfidf_index_vocab.json", "strategy_stats.toon"]:
        src = KNOWLEDGE_DIR / name
        if src.exists():
            dst = BACKUP_DIR / name
            shutil.copy2(src, dst)
            print(f"  Backed up: {name} -> backup/{name}")


def clean_records():
    valid = []
    skipped = 0
    total_bytes = 0

    with open(JSONL_FILE, "r") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            raw_size = len(line)
            total_bytes += raw_size
            if raw_size > MAX_RECORD_BYTES:
                print(f"  SKIP rec {i}: {raw_size:,} bytes (too large)")
                skipped += 1
                continue
            try:
                d = json.loads(line)
                exp_id = d.get("experiment_id", f"rec_{i}")
                rec_size = len(json.dumps(d))
                if rec_size > MAX_RECORD_BYTES:
                    print(f"  SKIP {exp_id}: parsed size {rec_size:,} bytes")
                    skipped += 1
                    continue
                valid.append(d)
            except json.JSONDecodeError as e:
                print(f"  SKIP rec {i}: JSON parse error: {e}")
                skipped += 1

    print(f"\n  Valid records: {len(valid)}")
    print(f"  Skipped records: {skipped}")
    print(f"  Original file: {total_bytes:,} bytes")
    return valid


def write_clean_jsonl(records):
    with open(JSONL_FILE, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    size = JSONL_FILE.stat().st_size
    print(f"  Wrote clean JSONL: {size:,} bytes")


def write_clean_toon(records):
    from toon_format import to_toon
    text = to_toon(records)
    TOON_FILE.write_text(text)
    size = TOON_FILE.stat().st_size
    print(f"  Wrote clean TOON: {size:,} bytes")


def rebuild_index(records):
    from tfidf_sparse import build_sparse_index, save_index

    indexed = []
    for rec in records:
        exp_id = rec.get("experiment_id")
        emb = rec.get("tfidf_embedding")
        if exp_id and emb:
            indexed.append({"experiment_id": exp_id, "tfidf_embedding": emb})

    if not indexed:
        print("  No records with tfidf_embedding, skipping index rebuild")
        return

    matrix, vocab, exp_ids = build_sparse_index(indexed)
    save_index(matrix, vocab, exp_ids)
    print(f"  Rebuilt index: {matrix.shape[0]} docs, {matrix.shape[1]} vocab terms")


def main():
    print("=== Knowledge Base Cleanup ===\n")

    if not JSONL_FILE.exists():
        print(f"ERROR: {JSONL_FILE} not found")
        return

    print("Step 1: Backing up files...")
    backup_files()

    print("\nStep 2: Cleaning records...")
    valid = clean_records()

    print("\nStep 3: Writing clean JSONL...")
    write_clean_jsonl(valid)

    print("\nStep 4: Writing clean TOON...")
    write_clean_toon(valid)

    print("\nStep 5: Rebuilding sparse index...")
    rebuild_index(valid)

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
