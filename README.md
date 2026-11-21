# AI-Powered Iterative MSA Pipeline

An intelligent multiple sequence alignment pipeline that uses ESM-2 embeddings, LLM-driven decision making, and a RAG-based learning system to iteratively improve protein alignments.

## Features

- **3-Tool Auto-Switch**: MAFFT ↔ MUSCLE ↔ ESM-2 alignment with intelligent switching
- **ESM-2 Integration**: Protein embeddings for alignment scoring and residue-level similarity
- **ESM Post-Alignment Analysis**: Optional per-residue probability/conservation analysis with gap refinement (`--esm-analysis`, `--esm-refine-gaps`, `--esm-alpha`); `--esm-post always|if_bad|off` controls post-correction after the main pipeline
- **Hill Numbers**: Multi-scale column diversity scoring (q0, q1, q2)
- **LLM Controller**: Ollama-powered AI agent for real-time alignment decisions
- **LLM resilience**: generation capped at 200 tokens (`num_predict`); if Ollama times out the pipeline falls back to a safe rule-based decision instead of failing the family
- **Per-tool flag validation**: LLM `extra_args` are whitelisted per tool (MAFFT flags can never leak into a MUSCLE run)
- **Trimal gentle-only**: Trim limited to `-gappyout` / `-automated1` (rotated). `-strict`/`-strictplus`/`-nogaps` are banned — they delete gap/indel columns
- **MUSCLE maxiters**: `-maxiters` passed to every MUSCLE run, clamped to ≥1000; MAFFT `--maxiterate` also clamped to ≥1000
- **RAG Learning**: Knowledge base stores successful experiments for warm start
- **ESM gap probing**: During learning, one ESM-2 try per family when gap ∈ [0.12, 0.40] (unlimited seqs); band success rates stored in `esm_gap_stats.toon.gz` and a learned “ESM works when gap > X” threshold feeds rules + the LLM prompt (`--tools` / `--dashboard` show the band table)
- **ESM post-correction learning**: Build and test both use `esm_post=if_bad` — refine only when gap still &gt;0.12 after the pipeline and the band is not learned-AVOID (empty stats → explore, so bad gaps still teach); `esm_post_stats.toon.gz` holds USE/AVOID/EXPLORE by band (`--tools` / `--dashboard`)
- **Warm start seeding**: Similar prior run (TF-IDF ≥ `-w`, default 0.8) seeds start tool/strategy (incl. direct ESM) and failed trim flags into the new run
- **Gzip knowledge base**: Experiments/stats stored as `.gz` via shell `gzip -c`/`gzip -dc` subprocess pipes
- **Sparse TF-IDF**: Efficient sequence similarity search for warm start matching
- **TOON Format**: Human-readable storage format for experiments and strategies
- **Batch Builder**: Safe knowledge base builder with signal handling, timeout enforcement, and gzip resume sidecar
- **Portable**: Works on any machine with path-independent scripts

## Requirements

- Python 3.11+
- Conda environments: `AI` (numpy, scipy, torch, transformers) and `seqTools` (mafft, muscle, trimal)
- Ollama with Llama3 model
- GPU recommended for ESM-2 inference (16GB+ VRAM)

## Installation

### Option 1: Quick Setup (Recommended)

```bash
# Clone or copy repository
cd AI

# Run setup script (creates conda environments from YAML files)
bash setup.sh        # Linux/Mac
setup.bat            # Windows
```

### Option 2: Manual Setup

```bash
# Create conda environments
conda env create -f envs/seqTools.yml
conda env create -f envs/ai.yml

# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3
```

## Usage

```bash
# Run pipeline on FASTA file
python run_pipeline.py input.fasta

# With options
python run_pipeline.py input.fasta -n 15 -t 0.02 -p 3 -w 0.85

# With ESM post-alignment analysis (conservation comparison + column report)
python run_pipeline.py input.fasta -n 10 --esm-analysis

# ESM analysis + gap refinement (alpha scales gap penalties by ESM conservation)
python run_pipeline.py input.fasta -n 10 --esm-analysis --esm-refine-gaps --esm-alpha 2.0

# Build knowledge base
./build.sh                    # wipe knowledge/+logs/ then full batch (PID → logs/batch_knowledge.pid)
./build.sh --resume           # keep KB, resume
./build.sh --limit 50         # extra args pass through
python scripts/build_knowledge.py --dry-run --limit 50

# Dashboard
python scripts/dashboard.py stats
python scripts/dashboard.py knowledge
python scripts/dashboard.py search "your query"
```

## Portable Drive Setup

The project can run from a portable drive on any machine:

### What Goes on Portable Drive

| Item | Size | Notes |
|------|------|-------|
| `scripts/` | ~1MB | All code (path-independent) |
| `data/pfam/families/` | ~11GB | 30,007 family files |
| `models/` | ~2.5GB | ESM-2 model |
| `envs/` | ~10KB | Conda YAML files |
| `knowledge/` | variable | Always stored as `.gz` (read/written via `gzip` subprocess) |
| Config files | ~10KB | README, setup scripts, etc. |

### What Stays Local

| Item | Why |
|------|-----|
| `knowledge/` | Active during experiments (experiments.jsonl.gz, strategy_stats.toon.gz, index) |
| `logs/` | Experiment logs |
| Conda environments | Created per-machine |

### Moving Between Machines

```bash
# 1. Copy portable drive to new machine
# 2. Setup conda environments
bash setup.sh

# 3. Run pipeline (knowledge base is always .gz — no manual (de)compression needed)
python run_pipeline.py input.fasta -n 10
```

## Project Structure

```
AI/
├── run_pipeline.py              # Main entry point
├── scripts/
│   ├── config.py                # Shared constants (CONDA_ENV, paths)
│   ├── pipeline_loop.py         # Iterative pipeline orchestrator
│   ├── ai_controller.py         # LLM decision engine
│   ├── msa_runner.py            # MAFFT runner
│   ├── muscle_runner.py         # MUSCLE runner
│   ├── esm_aligner.py           # ESM-2 alignment (NW + star)
│   ├── protein_embeddings.py    # ESM-2 embedding computation
│   ├── extended_metrics.py      # 14 metrics including Hill numbers
│   ├── esm_analysis.py          # ESM probability/conservation post-analysis
│   ├── rag_retrieve.py          # Warm start & knowledge retrieval (gzip subprocess)
│   ├── rag_store.py             # Knowledge base storage (gzip subprocess append)
│   ├── toon_format.py           # TOON encoder/decoder
│   ├── tfidf_sparse.py          # Sparse TF-IDF index
│   ├── embeddings.py            # TF-IDF embeddings
│   ├── convergence.py           # Convergence detection
│   ├── llm_core.py              # Ollama LLM interface
│   ├── build_knowledge.py       # Batch KB builder (safe, signal handling)
│   ├── pfamSep.py               # Pfam FASTA splitter
│   ├── dashboard.py             # CLI dashboard tool
│   ├── format_tools.py          # Storage inspection commands
│   ├── clean_kb.py              # Knowledge base cleanup tool
│   └── test_pipeline.py         # 22 tests
├── envs/
│   ├── seqTools.yml             # Conda env for alignment tools
│   └── ai.yml                   # Conda env for Python deps
├── knowledge/                   # Knowledge base (.gz JSONL/TOON + sparse index)
├── data/
│   ├── pfam/
│   │   ├── families/            # 30,007 family .faa files
│   │   └── sample/              # 50 random families for testing
│   └── README.txt               # Data placement instructions
├── models/
│   ├── models--facebook--esm2_t33_650M_UR50D/  # ESM-2 650M model
│   └── README.txt               # Model placement instructions
├── logs/                        # Build logs and experiment logs
├── trash/                       # Deprecated scripts
├── setup.sh                     # Linux/Mac setup script
├── setup.bat                    # Windows setup script
├── WORKSTATION_REQUIREMENTS.md  # Hardware specs
├── WORKSTATION_CONFIGURATIONS.tsv  # GPU comparison table
├── FlowChart.md                 # Full system flowcharts (Mermaid, 8 diagrams)
├── ALIGNMENT_SCORING.md         # Good/bad alignment scoring (Hill, Shannon, thresholds)
├── ContactPairMapping.md        # Contact-pair mapping ideas (protect contact regions from trim)
├── FutureUpdates.md             # Implementation roadmap
└── HANDOFF.md                   # Project handoff document
```

## Documentation

| Doc | What it covers |
|-----|----------------|
| [FlowChart.md](FlowChart.md) | **How the AI works end-to-end** — 8 Mermaid diagrams: big picture, loop, decision engine (rules → LLM → overrides), tools, ESM, knowledge/RAG, batch build, reporting |
| [ALIGNMENT_SCORING.md](ALIGNMENT_SCORING.md) | Good vs bad scoring: gap fraction, Shannon conservation, Hill q0/q1/q2, thresholds |
| [ContactPairMapping.md](ContactPairMapping.md) | Ideas for contact-pair maps (MI / ESM attention) to protect contact regions from trimming |
| [HANDOFF.md](HANDOFF.md) | Complete project handoff / architecture reference |
| [FutureUpdates.md](FutureUpdates.md) | Roadmap |

## Development

- **Concept & Architecture**: Human-designed (pipeline design, metric selection, tool integration strategy)
- **Implementation**: AI-assisted using OpenCode with MiMo v2.5 Free (`opencode/mimo-v2.5-free`)
- **Platform**: OpenCode CLI (https://opencode.ai)

## Metrics

The pipeline computes 14 alignment quality metrics (formulas and good/bad thresholds in [ALIGNMENT_SCORING.md](ALIGNMENT_SCORING.md)):
- Gap fraction, alignment length, sequence length
- SP-score, RMSE, sequence identity
- Column diversity (Shannon entropy, average product, and Hill numbers q0/q1/q2)
- Percentage of well-aligned columns

Full decision flow (rules, LLM, overrides) is diagrammed in [FlowChart.md](FlowChart.md).

## Knowledge Base Builder

The `build_knowledge.py` script safely processes Pfam families to populate the knowledge base:

- **Signal handling**: SIGINT/SIGTERM for graceful shutdown
- **Timeout enforcement**: Kills stuck pipelines after timeout
- **Child cleanup**: Kills mafft/muscle/trimal subprocesses on shutdown
- **Resume support**: Skips already-processed families via gzip sidecar (`knowledge/processed_families.txt.gz`)
- **Min sequences filter**: Skips families with < N sequences (default: 10)
- **Dry-run mode**: Preview families without processing

```bash
# Preview what would be processed
python scripts/build_knowledge.py --dry-run --limit 100

# Fresh full build (wipes knowledge/+logs/, records PID)
./build.sh

# Resume after interrupt / Ollama restart
./build.sh --resume

# Check status
ps -p $(cat logs/batch_knowledge.pid)
grep -E "OK:|FAILED:" logs/batch_knowledge.log | tail -10

# If FAILED lines show Ollama "Read timed out": restart Ollama, then resume
sudo systemctl restart ollama
./build.sh --resume
```

## License

Licensed under the Apache License, Version 2.0 — see [LICENSE](LICENSE).
