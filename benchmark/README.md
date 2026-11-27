# Benchmark — test the pipeline against BAliBASE-style databases

Scores the pipeline (and raw MAFFT/MUSCLE baselines) on reference alignment
sets from Edgar's [bench collection](https://www.drive5.com/bench/):
**BAliBASE v3, PREFAB4, OXBench, SABRE** — plus parameter tuning with Optuna.

## Quick start

```bash
bash benchmark/setup.sh          # env + data + unit test + 5-case smoke test
```

Then (after the knowledge batch is done, to avoid GPU contention):

```bash
cd benchmark

# Full benchmark
conda run -n bench python bench_runner.py --dataset bali3 --method all --resume

# Parameter tuning (random search)
conda run -n bench python opt_tune.py --sampler random --trials 30 --n-cases 15

# Parameter tuning (TPE / Bayesian)
conda run -n bench python opt_tune.py --sampler tpe --trials 50 --n-cases 15 --resume

# Report
conda run -n bench python report.py --corr
```

## What gets scored

| Group | Metrics | Source |
|-------|---------|--------|
| Ground truth | **SP**, **TC** vs reference | `sp_tc.py` (lowercase ref regions masked, qscore-style) |
| Our production metrics | `gap_fraction`, `sp_score`, `cs_score`, Shannon `conservation`, `gap_distribution`, `identity`, `hill_q0/q1/q2`… | `scripts/extended_metrics.py` (same code as the KB) |

## Layout

```
benchmark/
├── setup.sh           # one-shot setup (idempotent)
├── download_data.sh   # fetch bench.tar.gz → data/bench1.0/
├── env.yml            # conda env "bench" (numpy/pandas/optuna/…)
├── sp_tc.py           # SP/TC scorer + --selftest
├── pipeline_case.py   # runs pipeline under AI env, writes result JSON
├── bench_runner.py    # per-case loop, timeout+killpg, checkpointed JSONL
├── opt_tune.py        # Optuna (random|tpe) over pipeline params
├── report.py          # leaderboards + metric-vs-SP correlations
├── data/              # gitignored: bench1.0/, kb/ (isolated KB)
└── results/           # gitignored: results.jsonl, optuna.db, best_params.json
```

## Notes

- **Isolation**: bench runs set `MSA_KNOWLEDGE_DIR=benchmark/data/kb` and
  `MSA_LOGS_DIR=benchmark/results/run_logs` — the real `knowledge/` (and the
  running batch) is never touched.
- **Pipeline subprocesses** run under the `AI` env python (torch/ESM/Ollama);
  the harness itself runs under the `bench` env.
- **Resume**: `--resume` skips rows already in `results.jsonl`; Optuna study
  lives in `results/optuna.db` (also resumable).
- **Dataset note**: reference files use `.` as gap and lowercase for
  non-assessment regions — the scorer handles both.
- **Early finding (smoke, 5 BAliBASE cases)**: raw MAFFT/MUSCLE already score
  SP ≈ 0.6–1.0, while the pipeline's in-loop ESM probe (rule: gap ∈ [0.12,0.4]
  → try esm2) switches away from mafft and drops SP to ≈ 0.1–0.2 — the ESM
  star-aligner is much weaker than MAFFT on these sets. Tuning + the full run
  will quantify this; it is exactly the kind of regression the benchmark exists
  to catch.

See [PLAN.md](PLAN.md) for the full design.
