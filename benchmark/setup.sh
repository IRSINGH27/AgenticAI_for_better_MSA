#!/usr/bin/env bash
# Setup the benchmark: conda env "bench" + dataset download + smoke test.
# Safe to re-run (idempotent). Does NOT touch the running knowledge batch:
# bench isolates KB/logs via MSA_KNOWLEDGE_DIR / MSA_LOGS_DIR.
set -euo pipefail
cd "$(dirname "$0")"

echo "== [1/4] conda env 'bench' =="
if conda env list | awk '{print $1}' | grep -qx "bench"; then
    echo "env 'bench' exists — updating"
    conda env update -f env.yml --prune -q
else
    conda env create -f env.yml -q
fi

echo "== [2/4] dataset =="
bash download_data.sh

echo "== [3/4] scorer unit test =="
python3 sp_tc.py --selftest

echo "== [4/4] smoke test: 5 BAliBASE cases =="
# Baselines (bench env)
conda run -n bench python bench_runner.py \
    --dataset bali3 --method mafft --limit 5 --timeout 60 --resume

# Pipeline, short & GPU-light (esm-post off; no ESM analysis) to avoid
# disturbing anything else running on the GPU
conda run -n bench python bench_runner.py \
    --dataset bali3 --method pipeline --limit 5 --timeout 300 --resume \
    --pipeline-args '["-n","5","--esm-post","off"]'

echo ""
echo "== smoke report =="
conda run -n bench python report.py

echo ""
echo "Setup complete."
echo "  Full run:  conda run -n bench python bench_runner.py --dataset bali3 --method all --resume"
echo "  Tuning:    conda run -n bench python opt_tune.py --sampler random --trials 20"
