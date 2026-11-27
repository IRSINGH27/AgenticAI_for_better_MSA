#!/usr/bin/env python3
"""Benchmark runner: score pipeline / MAFFT / MUSCLE against reference alignments.

Each case: unaligned FASTA (data/<ds>/in/<id>) + reference (data/<ds>/ref/<id>).
Every method's alignment is scored with SP/TC (sp_tc.py) AND our production
metrics (scripts/extended_metrics.compute_extended_metrics).

Runs entirely via subprocesses (uniform timeout + killpg), so a wedged tool
never hangs the harness. Pipeline subprocesses run under the AI env python;
KB/logs are isolated via MSA_KNOWLEDGE_DIR / MSA_LOGS_DIR.

Usage:
  python bench_runner.py --dataset bali3 --method all --limit 5 --resume
  python bench_runner.py --dataset bali3 --method mafft --limit 5
Internal (spawned per case — do not call directly):
  python bench_runner.py --_single --method mafft --fasta X --out Y
"""
import argparse
import json
import os
import pathlib
import signal
import subprocess
import sys
import time
from typing import Dict, List, Optional

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
SCRIPTS = ROOT / "scripts"
DATA = HERE / "data" / "bench1.0"
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)
RESULTS_LOG = RESULTS / "results.jsonl"
KB_DIR = HERE / "data" / "kb"
RUN_LOGS = RESULTS / "run_logs"

sys.path.insert(0, str(HERE))          # sp_tc
sys.path.insert(0, str(SCRIPTS))       # extended_metrics, msa_runner, muscle_runner

import sp_tc  # noqa: E402

BENCH_PY = sys.executable
AI_PY = os.environ.get("BENCH_AI_PYTHON") or str(
    pathlib.Path.home() / "miniforge3/envs/AI/bin/python3"
)

DATASETS = ["bali3", "bali3pdb", "prefab4", "ox", "oxx", "sabre"]


def bench_env() -> Dict[str, str]:
    """Subprocess env with KB/logs isolation (never touches the real knowledge/)."""
    env = dict(os.environ)
    env["MSA_KNOWLEDGE_DIR"] = str(KB_DIR)
    env["MSA_LOGS_DIR"] = str(RUN_LOGS)
    return env


def run_subprocess(cmd: List[str], timeout: float, env: Optional[Dict[str, str]] = None) -> Dict:
    """Run cmd in its own process group; kill the whole group on timeout."""
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        env=env, start_new_session=True, text=True,
    )
    try:
        out, err = proc.communicate(timeout=timeout)
        return {"returncode": proc.returncode, "stdout": out, "stderr": err,
                "timed_out": False}
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            proc.kill()
        out, err = proc.communicate()
        return {"returncode": -9, "stdout": out or "", "stderr": err or "",
                "timed_out": True}


# ----------------------------------------------------------------- single case

def _single(method: str, fasta_path: str, ref_path: str, out_path: str,
            timeout: float, pipeline_args: List[str], no_mask: bool) -> int:
    """Align one case (this process IS the per-case worker) + score + write JSON."""
    from extended_metrics import compute_extended_metrics

    fasta_text = pathlib.Path(fasta_path).read_text()
    t0 = time.time()
    error = None
    alignment = ""
    extra: Dict = {}

    try:
        if method == "mafft":
            from msa_runner import run_msa
            alignment = run_msa(fasta_text, {"strategy": "--auto"})
        elif method == "muscle":
            from muscle_runner import run_muscle
            alignment = run_muscle(fasta_text, {"algorithm": "align"})
        elif method == "pipeline":
            # Re-exec ourselves under the AI env for the full pipeline
            tmp_out = out_path + ".pipe.json"
            cmd = [AI_PY, str(HERE / "pipeline_case.py"),
                   "--fasta", fasta_path, "--out", tmp_out] + pipeline_args
            r = run_subprocess(cmd, timeout=timeout, env=bench_env())
            if r["timed_out"]:
                raise TimeoutError(f"pipeline exceeded {timeout}s")
            if r["returncode"] != 0:
                raise RuntimeError(f"pipeline failed rc={r['returncode']}: "
                                   f"{r['stderr'][-500:]}")
            payload = json.loads(pathlib.Path(tmp_out).read_text())
            alignment = payload.pop("alignment")
            extra = payload
        else:
            raise ValueError(f"unknown method {method}")
    except Exception as e:  # noqa: BLE001 — record failure, keep going
        error = f"{type(e).__name__}: {e}"

    runtime = round(time.time() - t0, 2)
    row: Dict = {
        "dataset": None, "case": pathlib.Path(fasta_path).name,
        "method": method, "error": error, "runtime_s": runtime,
    }
    if alignment and not error:
        ref_text = pathlib.Path(ref_path).read_text()
        row.update(sp_tc.score(alignment, ref_text, mask_lowercase=not no_mask))
        row["extended_metrics"] = compute_extended_metrics(alignment)
        row.update({k: extra.get(k) for k in
                    ("iterations", "experiment_id", "was_trimmed",
                     "esm_post_applied") if k in extra})
    pathlib.Path(out_path).write_text(json.dumps(row))
    return 0 if not error else 1


# ----------------------------------------------------------------- dataset scan

def list_cases(dataset: str) -> List[str]:
    ids_file = DATA / dataset / "info" / "ids.txt"
    if ids_file.exists():
        return [l.strip() for l in ids_file.read_text().splitlines() if l.strip()]
    in_dir = DATA / dataset / "in"
    return sorted(p.name for p in in_dir.iterdir() if p.is_file())


def make_row_key(row: Dict) -> tuple:
    return (row.get("dataset"), row.get("case"), row.get("method"))


def load_done(path: pathlib.Path = RESULTS_LOG) -> set:
    done = set()
    if path.exists():
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                done.add(make_row_key(json.loads(line)))
            except json.JSONDecodeError:
                continue
    return done


# ----------------------------------------------------------------- main loop

def run_bench(args) -> int:
    cases = list_cases(args.dataset)
    if args.seed_cases:
        import random
        random.Random(args.seed_cases).shuffle(cases)
    if args.limit:
        cases = cases[: args.limit]

    methods = (["mafft", "muscle", "pipeline"] if args.method == "all"
               else [args.method])
    done = load_done() if args.resume else set()

    in_dir = DATA / args.dataset / "in"
    ref_dir = DATA / args.dataset / "ref"
    n_ok = n_fail = n_skip = 0

    for case in cases:
        fasta = in_dir / case
        ref = ref_dir / case
        if not fasta.exists() or not ref.exists():
            print(f"  SKIP {case} (missing in/ref)")
            n_skip += 1
            continue
        for method in methods:
            key = (args.dataset, case, method)
            if key in done:
                n_skip += 1
                continue
            tmp_out = RESULTS / f".tmp_{args.dataset}_{case}_{method}.json"
            cmd = [BENCH_PY, str(pathlib.Path(__file__).resolve()),
                   "--_single", "--method", method,
                   "--fasta", str(fasta), "--ref", str(ref),
                   "--out", str(tmp_out),
                   "--timeout", str(args.timeout),
                   "--pipeline-args", args.pipeline_args,
                   ] + (["--no-mask"] if args.no_mask else [])
            r = run_subprocess(cmd, timeout=args.timeout + 30, env=bench_env())
            if tmp_out.exists():
                row = json.loads(tmp_out.read_text())
                row["dataset"] = args.dataset
                tmp_out.unlink()
                with RESULTS_LOG.open("a") as f:
                    f.write(json.dumps(row) + "\n")
                if row.get("error"):
                    n_fail += 1
                    print(f"  FAIL {args.dataset}/{case}/{method}: {row['error'][:120]}")
                else:
                    n_ok += 1
                    print(f"  ok   {args.dataset}/{case}/{method}: "
                          f"SP={row['sp']:.3f} TC={row['tc']:.3f} "
                          f"gap={row['extended_metrics']['gap_fraction']:.3f} "
                          f"({row['runtime_s']}s)")
            else:
                n_fail += 1
                print(f"  FAIL {args.dataset}/{case}/{method}: no output "
                      f"rc={r['returncode']} {r['stderr'][-200:]}")

    print(f"\nDone: {n_ok} ok, {n_fail} failed, {n_skip} skipped/cached "
          f"→ {RESULTS_LOG}")
    return 0 if n_fail == 0 else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=DATASETS, default="bali3")
    ap.add_argument("--method", choices=["pipeline", "mafft", "muscle", "all"],
                    default="all")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--timeout", type=float, default=120.0,
                    help="per-case seconds")
    ap.add_argument("--resume", action="store_true",
                    help="skip cases already in results.jsonl")
    ap.add_argument("--no-mask", action="store_true",
                    help="score lowercase reference regions too")
    ap.add_argument("--seed-cases", type=int, default=None,
                    help="shuffle case order with seed (for fair subsampling)")
    ap.add_argument("--pipeline-args", default="",
                    help="extra args for pipeline_case.py (JSON list string)")
    # internal single-case mode
    ap.add_argument("--_single", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--fasta", help=argparse.SUPPRESS)
    ap.add_argument("--ref", help=argparse.SUPPRESS)
    ap.add_argument("--out", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args._single:
        import json as _json
        pargs = _json.loads(args.pipeline_args) if args.pipeline_args else []
        return _single(args.method, args.fasta, args.ref, args.out,
                       args.timeout, pargs, args.no_mask)
    return run_bench(args)


if __name__ == "__main__":
    sys.exit(main())
