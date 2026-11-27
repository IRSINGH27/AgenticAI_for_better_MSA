#!/usr/bin/env python3
"""Optuna parameter tuner for the MSA pipeline.

Objective: mean SP (vs reference) over N sampled benchmark cases per trial.
Our production metrics (gap_fraction, hill_q1, conservation) are logged as
trial attributes so we can see how tuned params shift them.

Samplers: random | tpe   Storage: results/optuna.db (resumable).

Usage:
  python opt_tune.py --sampler random --trials 20 --n-cases 10 --dataset bali3
  python opt_tune.py --sampler tpe --trials 50 --n-cases 20 --dataset bali3 --resume
"""
import argparse
import json
import pathlib
import random
import sys
import time
from typing import Dict, List

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import sp_tc  # noqa: E402

RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)
DB_PATH = RESULTS / "optuna.db"
BEST_PATH = RESULTS / "best_params.json"
INCOMPLETE = RESULTS / ".trial_tmp.json"

import optuna  # noqa: E402
from bench_runner import (  # noqa: E402
    AI_PY, BENCH_PY, DATA, RESULTS_LOG, bench_env, list_cases, run_subprocess,
)


def sample_cases(dataset: str, n: int, seed: int) -> List[str]:
    cases = list_cases(dataset)
    # only cases with both in/ and ref/
    cases = [c for c in cases
             if (DATA / dataset / "in" / c).exists()
             and (DATA / dataset / "ref" / c).exists()]
    rng = random.Random(seed)
    if n >= len(cases):
        rng.shuffle(cases)
        return cases
    return rng.sample(cases, n)


def run_one(dataset: str, case: str, params: Dict, timeout: float) -> Dict:
    """Run pipeline on one case via subprocess; return scored row (or error)."""
    fasta = DATA / dataset / "in" / case
    ref = DATA / dataset / "ref" / case
    tmp_out = RESULTS / f".tune_{case}.json"
    if tmp_out.exists():
        tmp_out.unlink()

    pargs: List[str] = [
        "-n", str(params["max_iterations"]),
        "-t", str(params["tolerance"]),
        "-p", str(params["patience"]),
        "-w", str(params["warm_start"]),
        "--esm-post", params["esm_post"],
        "--esm-alpha", str(params["esm_alpha"]),
    ]
    if params["esm_analysis"]:
        pargs.append("--esm-analysis")

    cmd = [BENCH_PY, str(HERE / "bench_runner.py"),
           "--_single", "--method", "pipeline",
           "--fasta", str(fasta), "--ref", str(ref),
           "--out", str(tmp_out), "--timeout", str(timeout),
           "--pipeline-args", json.dumps(pargs)]
    run_subprocess(cmd, timeout=timeout + 30, env=bench_env())
    if not tmp_out.exists():
        return {"error": "no output (timeout or crash)", "sp": 0.0, "tc": 0.0}
    row = json.loads(tmp_out.read_text())
    tmp_out.unlink(missing_ok=True)
    return row


def make_objective(dataset: str, n_cases: int, seed: int, timeout: float,
                   trial_seed: int):
    def objective(trial: optuna.Trial) -> float:
        params = {
            "max_iterations": trial.suggest_int("max_iterations", 5, 50),
            "tolerance": trial.suggest_float("tolerance", 1e-4, 0.05, log=True),
            "patience": trial.suggest_int("patience", 1, 5),
            "warm_start": trial.suggest_float("warm_start", 0.5, 0.95),
            "esm_post": trial.suggest_categorical(
                "esm_post", ["always", "if_bad", "off"]),
            "esm_alpha": trial.suggest_float("esm_alpha", 0.5, 4.0),
            "esm_analysis": trial.suggest_categorical(
                "esm_analysis", [False, True]),
        }
        cases = sample_cases(dataset, n_cases, seed + trial.number)
        sps, tcs, gaps, hills, errs = [], [], [], [], 0
        t0 = time.time()
        for case in cases:
            row = run_one(dataset, case, params, timeout)
            if row.get("error"):
                errs += 1
                continue
            sps.append(row["sp"])
            tcs.append(row["tc"])
            em = row.get("extended_metrics") or {}
            if "gap_fraction" in em:
                gaps.append(em["gap_fraction"])
            if "hill_q1_avg" in em:
                hills.append(em["hill_q1_avg"])
        if not sps:
            raise optuna.TrialPruned()
        mean_sp = sum(sps) / len(sps)
        trial.set_user_attr("mean_tc", sum(tcs) / len(tcs))
        trial.set_user_attr("mean_gap", sum(gaps) / len(gaps) if gaps else None)
        trial.set_user_attr("mean_hill_q1",
                            sum(hills) / len(hills) if hills else None)
        trial.set_user_attr("n_cases", len(sps))
        trial.set_user_attr("n_errors", errs)
        trial.set_user_attr("params", params)
        trial.set_user_attr("runtime_s", round(time.time() - t0, 1))
        # SP primary, TC tiebreak (small weight)
        return mean_sp + 0.01 * trial.user_attrs["mean_tc"]
    return objective


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="bali3",
                    choices=["bali3", "bali3pdb", "prefab4", "ox", "oxx", "sabre"])
    ap.add_argument("--sampler", choices=["random", "tpe"], default="random")
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--n-cases", type=int, default=10,
                    help="benchmark cases evaluated per trial")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--resume", action="store_true",
                    help="continue existing study in optuna.db")
    ap.add_argument("--study-name", default="msa_tune")
    args = ap.parse_args()

    storage = f"sqlite:///{DB_PATH}"
    sampler = (optuna.samplers.RandomSampler(seed=args.seed)
               if args.sampler == "random"
               else optuna.samplers.TPESampler(seed=args.seed))
    study = optuna.create_study(
        study_name=f"{args.study_name}_{args.dataset}_{args.sampler}",
        storage=storage, load_if_exists=args.resume,
        direction="maximize", sampler=sampler,
    )

    objective = make_objective(args.dataset, args.n_cases, args.seed,
                               args.timeout, args.seed)
    study.optimize(objective, n_trials=args.trials, show_progress_bar=True)

    best = study.best_trial
    result = {
        "dataset": args.dataset,
        "sampler": args.sampler,
        "best_value": round(best.value, 6),
        "best_params": best.params,
        "best_attrs": {k: best.user_attrs.get(k)
                       for k in ("mean_tc", "mean_gap", "mean_hill_q1",
                                 "n_cases", "runtime_s")},
        "n_trials": len(study.trials),
    }
    BEST_PATH.write_text(json.dumps(result, indent=2))

    print("\n=== BEST TRIAL ===")
    print(json.dumps(result, indent=2))
    try:
        imp = optuna.importance.get_param_importances(study)
        print("\n=== PARAM IMPORTANCE ===")
        for k, v in imp.items():
            print(f"  {k:20s} {v:.4f}")
    except Exception as e:  # noqa: BLE001
        print(f"(importance unavailable: {e})")
    print(f"\nSaved → {BEST_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
