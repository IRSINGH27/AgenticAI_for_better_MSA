#!/usr/bin/env python3
"""
Batch knowledge base builder.
Runs MSA pipeline on Pfam families to populate the knowledge base.

Safe version with signal handling, timeout enforcement, and child cleanup.

Usage:
    python scripts/build_knowledge.py --limit 100 --max-seqs 8000
    python scripts/build_knowledge.py --dry-run --limit 10
    python scripts/build_knowledge.py --timeout 600 --max-seqs 4000
"""

import os
import sys
import random
import argparse
import pathlib
import time
import signal
import subprocess
import multiprocessing
from datetime import datetime
from typing import List, Tuple, Optional

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from rag_store import KNOWLEDGE_DIR

PROCESSED_FILE = "processed_families.txt.gz"

# Global shutdown flag
_shutdown_requested = False
_current_process = None


def signal_handler(signum, frame):
    """Handle signals gracefully."""
    global _shutdown_requested, _current_process

    sig_name = signal.Signals(signum).name
    print(f"\n[ signal: {sig_name} ] Requesting graceful shutdown... (Ctrl+C again to force)")
    _shutdown_requested = True

    # If a child process is running, kill its process group
    if _current_process is not None and _current_process.is_alive():
        try:
            os.killpg(os.getpgid(_current_process.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass

    # Second signal = force kill
    signal.signal(signal.SIGINT, force_handler)
    signal.signal(signal.SIGTERM, force_handler)


def force_handler(signum, frame):
    """Force kill on second signal."""
    global _current_process
    print("\n[ force kill ] Terminating immediately...")
    if _current_process is not None and _current_process.is_alive():
        try:
            os.killpg(os.getpgid(_current_process.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    sys.exit(1)


def setup_signals():
    """Register signal handlers."""
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    # Ignore SIGTSTP (Ctrl+Z) - don't suspend
    signal.signal(signal.SIGTSTP, signal.SIG_IGN)


def get_family_files(families_dir: str) -> List[Tuple[str, int, str]]:
    """Get family FASTA files sorted by size (ascending).
    Returns list of (filename, seq_count, full_path).
    """
    families = []
    families_path = pathlib.Path(families_dir)

    for f in families_path.glob("*.faa"):
        try:
            seq_count = count_sequences(str(f))
            families.append((f.name, seq_count, str(f)))
        except Exception:
            continue

    families.sort(key=lambda x: x[1])
    return families


def count_sequences(fasta_path: str) -> int:
    """Count sequences in a FASTA file."""
    count = 0
    with open(fasta_path, "r") as f:
        for line in f:
            if line.startswith(">"):
                count += 1
    return count


def read_fasta(fasta_path: str) -> str:
    """Read FASTA file contents."""
    with open(fasta_path, "r") as f:
        return f.read()


def subsample_fasta(fasta_text: str, max_seqs: int, seed: int = 42) -> str:
    """Randomly subsample sequences from FASTA text."""
    entries = []
    current_header = None
    current_seq = []

    for line in fasta_text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if current_header is not None:
                entries.append((current_header, "".join(current_seq)))
            current_header = line
            current_seq = []
        else:
            current_seq.append(line)

    if current_header is not None:
        entries.append((current_header, "".join(current_seq)))

    if len(entries) <= max_seqs:
        return fasta_text

    random.seed(seed)
    sampled = random.sample(entries, max_seqs)

    result = []
    for header, seq in sampled:
        result.append(header)
        for i in range(0, len(seq), 80):
            result.append(seq[i:i + 80])

    return "\n".join(result) + "\n"


def get_processed_families() -> set:
    """Get set of already-processed family names (gzip sidecar in knowledge/)."""
    path = KNOWLEDGE_DIR / PROCESSED_FILE
    processed = set()
    if path.exists():
        # check=False: multi-member appends can leave trailing garbage (exit 2)
        # but still decompress the valid members we need.
        out = subprocess.run(["gzip", "-dc", str(path)], capture_output=True, check=False)
        if out.returncode not in (0, 2) and not out.stdout:
            raise RuntimeError(f"Cannot read {path}: {out.stderr.decode(errors='replace')}")
        for line in out.stdout.decode("utf-8").splitlines():
            line = line.strip()
            if line:
                processed.add(line)
    return processed


def mark_family_processed(filename: str):
    """Append family name to the gzip sidecar (one gzip member per line)."""
    KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
    path = KNOWLEDGE_DIR / PROCESSED_FILE
    name = filename.replace(".faa", "")
    with open(path, "ab") as f:
        subprocess.run(["gzip", "-c"], input=(name + "\n").encode("utf-8"), stdout=f, check=True)


def setup_logging(log_path: str):
    """Setup log file for batch processing."""
    log_dir = pathlib.Path(log_path).parent
    log_dir.mkdir(parents=True, exist_ok=True)


def log_message(msg: str, log_path: str, also_print: bool = True):
    """Write to log file and optionally print. Always flushes."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    if also_print:
        print(line, flush=True)
    with open(log_path, "a") as f:
        f.write(line + "\n")
        f.flush()


def run_pipeline_in_process(fasta_text, max_iterations, convergence_tolerance,
                            convergence_patience, warm_start_threshold, result_dict,
                            esm_post="if_bad"):
    """Run pipeline in a separate process with its own process group."""
    # Create new process group so we can kill all children
    os.setpgrp()

    try:
        from pipeline_loop import run_pipeline
        result = run_pipeline(
            fasta_text=fasta_text,
            max_iterations=max_iterations,
            convergence_tolerance=convergence_tolerance,
            convergence_patience=convergence_patience,
            warm_start_threshold=warm_start_threshold,
            esm_post=esm_post,
        )
        result_dict.update(result)
        result_dict["__success"] = True
    except Exception as e:
        result_dict["__error"] = str(e)
        result_dict["__success"] = False
    finally:
        # Cleanup any child processes (mafft, muscle, trimal)
        try:
            os.killpg(os.getpgrp(), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass


def run_pipeline_with_timeout(fasta_text, timeout, max_iterations=100,
                              convergence_tolerance=0.001, convergence_patience=3,
                              warm_start_threshold=0.8, esm_post="if_bad"):
    """Run pipeline with timeout enforcement using multiprocessing."""
    global _current_process

    manager = multiprocessing.Manager()
    result_dict = manager.dict()

    p = multiprocessing.Process(
        target=run_pipeline_in_process,
        args=(fasta_text, max_iterations, convergence_tolerance,
              convergence_patience, warm_start_threshold, result_dict, esm_post)
    )
    _current_process = p
    p.start()

    p.join(timeout=timeout)

    if p.is_alive():
        # Timeout - kill the process group
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        p.join(timeout=5)
        _current_process = None
        raise TimeoutError(f"Pipeline timed out after {timeout}s")

    _current_process = None

    if not result_dict.get("__success", False):
        error = result_dict.get("__error")
        if error is None:
            # Child died without running except (signal/segfault) — report exit code
            code = p.exitcode
            if code is None:
                error = "child died with no result and no exit code"
            elif code < 0:
                error = f"child killed by signal {-code}" + (" (SIGSEGV — likely torch/CUDA)" if code == -11 else "")
            else:
                error = f"child exited with code {code} before setting result"
        raise RuntimeError(f"Pipeline failed: {error}")

    return dict(result_dict)


def main():
    global _shutdown_requested

    parser = argparse.ArgumentParser(
        description="Batch knowledge base builder for MSA pipeline"
    )
    parser.add_argument(
        "--families-dir",
        default=str(PROJECT_ROOT / "data" / "pfam" / "families"),
        help="Directory containing family FASTA files"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of families to process (default: all)"
    )
    parser.add_argument(
        "--max-seqs",
        type=int,
        default=8000,
        help="Max sequences per family (subsample if larger)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Timeout per family in seconds (default: 300)"
    )
    parser.add_argument(
        "--log-file",
        default=str(PROJECT_ROOT / "logs" / "batch_knowledge.log"),
        help="Log file path"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for subsampling"
    )
    parser.add_argument(
        "--min-seqs",
        type=int,
        default=10,
        help="Skip families with fewer sequences (default: 10)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview families without processing"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip families already in knowledge base"
    )
    args = parser.parse_args()

    setup_signals()
    setup_logging(args.log_file)

    # Get all families
    print(f"Scanning {args.families_dir}...", flush=True)
    families = get_family_files(args.families_dir)
    total_families = len(families)
    print(f"Found {total_families} families", flush=True)

    # Apply limit
    if args.limit is not None and args.limit > 0:
        families = families[:args.limit]
    print(f"Processing {len(families)} families (limit={args.limit or 'all'})", flush=True)

    # Resume: skip already processed
    if args.resume:
        processed = get_processed_families()
        before = len(families)
        families = [(f, c, p) for f, c, p in families if f.replace(".faa", "") not in processed]
        print(f"Resume: skipped {before - len(families)} already processed", flush=True)

    if args.dry_run:
        print(f"\n{'='*50}")
        print(f"DRY RUN - Would process {len(families)} families:")
        for i, (filename, seq_count, filepath) in enumerate(families[:20]):
            print(f"  [{i+1}] {filename}: {seq_count} seqs")
        if len(families) > 20:
            print(f"  ... and {len(families) - 20} more")
        print(f"{'='*50}")
        return

    # Stats
    stats = {
        "processed": 0,
        "skipped": 0,
        "failed": 0,
        "total_seqs": 0,
        "start_time": time.time()
    }

    log_message(f"=== Batch knowledge build started ===", args.log_file)
    log_message(f"Families to process: {len(families)}", args.log_file)
    log_message(f"Max sequences per family: {args.max_seqs}", args.log_file)
    log_message(f"Min sequences per family: {args.min_seqs}", args.log_file)
    log_message(f"Timeout per family: {args.timeout}s", args.log_file)

    for i, (filename, seq_count, filepath) in enumerate(families):
        # Check shutdown flag
        if _shutdown_requested:
            log_message(f"\n=== Shutdown requested ===", args.log_file)
            break

        # Skip small families
        if seq_count < args.min_seqs:
            log_message(
                f"  SKIP: {filename} ({seq_count} seqs < {args.min_seqs} min)",
                args.log_file
            )
            stats["skipped"] += 1
            continue

        # Progress
        elapsed = time.time() - stats["start_time"]
        rate = stats["processed"] / max(elapsed, 1)
        eta = (len(families) - i) / max(rate, 0.001) / 60

        print(f"\n[{i+1}/{len(families)}] {filename}: {seq_count} seqs (ETA: {eta:.0f}min)", flush=True)

        # Read and subsample
        try:
            fasta_text = read_fasta(filepath)
        except Exception as e:
            log_message(f"  FAILED to read: {filename} — {e}", args.log_file)
            stats["failed"] += 1
            continue

        if seq_count > args.max_seqs:
            print(f"  Subsampling {seq_count} -> {args.max_seqs}", flush=True)
            fasta_text = subsample_fasta(fasta_text, args.max_seqs, seed=args.seed + i)
            seq_count = args.max_seqs

        # Run pipeline with timeout
        try:
            result = run_pipeline_with_timeout(
                fasta_text=fasta_text,
                timeout=args.timeout,
                max_iterations=100,
                convergence_tolerance=0.001,
                convergence_patience=3,
                warm_start_threshold=0.8,
                esm_post="if_bad",
            )

            gap = result.get("final_metrics", {}).get("gap_fraction", -1)
            iters = result.get("iterations", 0)
            exp_id = result.get("experiment_id", "?")

            log_message(
                f"  OK: {filename} → {exp_id}, {iters} iters, gap={gap:.4f}",
                args.log_file
            )
            stats["processed"] += 1
            stats["total_seqs"] += seq_count
            mark_family_processed(filename)

        except TimeoutError:
            log_message(
                f"  TIMEOUT: {filename} ({seq_count} seqs, {args.timeout}s limit) — "
                f"pipeline exceeded {args.timeout}s, skipped",
                args.log_file
            )
            stats["failed"] += 1

        except Exception as e:
            log_message(
                f"  FAILED: {filename} — {type(e).__name__}: {e}",
                args.log_file
            )
            stats["failed"] += 1

    # Summary
    elapsed = time.time() - stats["start_time"]
    log_message(f"\n=== Batch complete ===", args.log_file)
    log_message(f"Processed: {stats['processed']}/{len(families)}", args.log_file)
    log_message(f"Skipped: {stats['skipped']} (too few seqs)", args.log_file)
    log_message(f"Failed: {stats['failed']}", args.log_file)
    log_message(f"Total sequences: {stats['total_seqs']:,}", args.log_file)
    log_message(f"Time: {elapsed/60:.1f} minutes", args.log_file)

    print(f"\n{'='*40}", flush=True)
    print(f"BATCH COMPLETE", flush=True)
    print(f"  Processed: {stats['processed']}", flush=True)
    print(f"  Skipped: {stats['skipped']} (< {args.min_seqs} seqs)", flush=True)
    print(f"  Failed: {stats['failed']}", flush=True)
    print(f"  Total seqs: {stats['total_seqs']:,}", flush=True)
    print(f"  Time: {elapsed/60:.1f} min", flush=True)


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    main()
