# Benchmark & Parameter Tuning Plan

Goal: test the MSA pipeline against BAliBASE-style reference databases and tune
pipeline parameters (random / TPE search), scored with both ground-truth
reference metrics and our own metric suite.

## Decisions (locked)

- **Dataset**: Edgar bench.tar.gz — BAliBASE v3 + PREFAB4 + OXBench + SABRE (FASTA)
- **Tuner**: Optuna (samplers: `random`, `tpe`) in a new conda env `bench`
- **Baselines**: raw MAFFT and raw MUSCLE scored on every case alongside the pipeline
- **Timing**: build + smoke test now; full runs after the batch finishes (GPU contention)
- **Git**: code committed; `benchmark/data/` and `benchmark/results/` gitignored

## Folder layout

```
benchmark/
├── PLAN.md               # this document
├── README.md             # how to run
├── env.yml               # conda env "bench"
├── setup.sh              # create env + download data + smoke test
├── download_data.sh      # fetch bench.tar.gz → data/
├── sp_tc.py              # SP + TC scorer vs reference alignment (+ unit test)
├── bench_runner.py       # run pipeline / mafft / muscle → score each case
├── opt_tune.py           # Optuna search (random / tpe) over pipeline params
├── report.py             # tables + plots (SP/TC + our metrics + correlations)
├── data/                 # GITIGNORED — balibase/, prefab/, oxbench/, sabre/
└── results/              # GITIGNORED — results.jsonl, optuna.db, best_params.json
```

## Conda env `bench` (env.yml)

- python 3.11, numpy, scipy, pandas, matplotlib, scikit-learn, pip → `optuna`
- Harness runs in `bench`; **pipeline calls shell out** to the `AI` env python
  (same pattern as mafft → seqTools). No torch/GPU duplication; Ollama/ESM stay in AI.

## Scoring

### 1. Reference-based (ground truth) — `sp_tc.py`
- Parse reference + predicted alignment into residue-pair sets (sequences matched by ID)
- **SP** = correct residue pairs / total reference pairs
- **TC** = fully correct columns / total reference columns
- Unit test on a hand-made tiny case (known SP/TC)

### 2. Our metrics (same as production KB) — reuse `scripts/extended_metrics.py`
- `compute_extended_metrics()` imported via `sys.path` — always in sync with production
- Per alignment: `gap_fraction`, internal `sp_score`, `cs_score`, Shannon
  `conservation`, `gap_distribution`, `identity`, `hill_q0/q1/q2_avg`,
  `hill_q1_max`, `hill_problematic_columns`
- Applied to **pipeline result AND MAFFT/MUSCLE baselines**

### 3. Results row (results.jsonl)
```
dataset, case, method, params, SP, TC, <full extended_metrics dict>,
iterations, runtime_s, experiment_id, timestamp
```

## Runner — `bench_runner.py`

- Input: dataset dir (unaligned FASTA + reference alignment pairs)
- Per case: unaligned FASTA → one of {pipeline, mafft, muscle} → score SP/TC + our metrics
- Per-case timeout (default 120s), checkpoint JSONL (resume-safe, `--resume`)
- **KB isolation**: `scripts/config.py` respects env vars
  `MSA_KNOWLEDGE_DIR` / `MSA_LOGS_DIR` (default behavior unchanged);
  bench points them at `benchmark/data/kb/` so tuning never pollutes the real KB
- CLI: `--dataset balibase --method pipeline|mafft|muscle|all --limit N --resume`

## Tuner — `opt_tune.py`

- **Samplers**: `random`, `tpe` (Optuna; sqlite storage in results/, resumable)
- **Search space**:

  | Param | Range |
  |-------|-------|
  | max_iterations | 5–50 |
  | tolerance | 1e-4–0.05 (log) |
  | patience | 1–5 |
  | warm_start_threshold | 0.5–0.95 |
  | esm_post | always / if_bad / off |
  | esm_alpha | 0.5–4.0 |
  | esm_analysis | on / off |

- **Objective**: mean SP over N cases per trial (default 20, seeded subsample);
  TC as tiebreak
- Our metrics logged as trial attributes (see how tuned params shift gap/hill)
- Outputs: `results/best_params.json`, Optuna importance ranking, trial CSV

## Report — `report.py`

- Leaderboard: method × dataset → mean SP/TC + mean gap_fraction / hill_q1 / conservation
- **Correlation views**: does our `gap_fraction`/`hill_q1` track reference SP?
  (validates our scoring theory against ground truth)
- Param importance bar chart (from Optuna)

## Setup — `setup.sh`

1. `conda env create -f benchmark/env.yml` (idempotent)
2. `download_data.sh`: wget bench.tar.gz → extract into `benchmark/data/`
3. Smoke test:
   - `sp_tc.py` unit test
   - 5 BAliBASE cases with MAFFT baseline
   - 5 BAliBASE cases with pipeline (short: `-n 5`, `esm_post off` to spare GPU)

## Git

- Commit: all `benchmark/*.py`, `env.yml`, `*.sh`, `PLAN.md`, `README.md`, config.py patch
- Ignore: `benchmark/data/`, `benchmark/results/`

## Execution order

1. Patch `scripts/config.py` (env-var override, 3 lines)
2. Write `benchmark/` files
3. Run `bash benchmark/setup.sh` → env + data + smoke test
4. Report smoke results → commit + push
5. Full benchmark + tuning **after the batch finishes**
