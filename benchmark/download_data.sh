#!/usr/bin/env bash
# Download Edgar's bench collection (BAliBASE v3, PREFAB4, OXBench, SABRE) — FASTA format.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p data
URL="https://www.drive5.com/bench/bench.tar.gz"
if [ -d data/bench1.0 ]; then
    echo "data/bench1.0 already present — skipping download."
    exit 0
fi
if [ ! -f data/bench.tar.gz ]; then
    echo "Downloading $URL ..."
    curl -sL --fail "$URL" -o data/bench.tar.gz
fi
echo "Extracting ..."
tar -xzf data/bench.tar.gz -C data/
echo "Done. Datasets:"
ls data/bench1.0/
