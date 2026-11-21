#!/bin/bash
# Setup script for MSA AI Pipeline
# Creates conda environments from YAML files

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENVS_DIR="$SCRIPT_DIR/envs"

echo "=========================================="
echo "MSA AI Pipeline Setup"
echo "=========================================="
echo ""

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda not found"
    echo "Install miniforge3: https://github.com/conda-forge/miniforge"
    exit 1
fi

# Create seqTools environment
echo "[1/2] Creating seqTools environment..."
if conda env list | grep -q "^seqTools "; then
    echo "  seqTools environment already exists"
else
    conda env create -f "$ENVS_DIR/seqTools.yml"
    echo "  seqTools environment created"
fi

# Create AI environment
echo "[2/2] Creating AI environment..."
if conda env list | grep -q "^AI "; then
    echo "  AI environment already exists"
else
    conda env create -f "$ENVS_DIR/ai.yml"
    echo "  AI environment created"
fi

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "Verify installation:"
echo "  conda run -n seqTools mafft --version"
echo "  conda run -n seqTools muscle --version"
echo "  conda run -n seqTools trimal --version"
echo "  conda run -n AI python -c 'import torch; print(torch.__version__)'"
echo ""
