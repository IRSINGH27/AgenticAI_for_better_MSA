# MSA AI Pipeline - Complete Project Handoff

**Last Updated:** November 11, 2026 (v1.5)
**Project Location:** `/home/user/AI/`
**Status:** Ready for fresh full KB build — gentle trims only (indels kept), ESM threshold ignores gap&lt;0.10 bands, `esm_post=if_bad`, `build.sh` launcher (wipe or resume)

---

## 1. Project Goal

Build an **AI-powered iterative Multiple Sequence Alignment (MSA) pipeline** that:
- Runs MAFFT, MUSCLE, and trimal alignment tools from Python
- Uses a local LLM (Ollama) to make intelligent decisions about which tool/parameters to use
- Learns from past experiments via a RAG (Retrieval-Augmented Generation) layer
- Improves alignment quality over time by recommending strategies that worked
- Evaluates alignment quality using multiple metrics
- Detects and skips duplicate experiments
- Exports/imports knowledge for sharing
- Processes Pfam protein families to build a comprehensive knowledge base

---

## 2. Architecture

```
User FASTA File
       ↓
[run_pipeline.py] (entry point)
       ↓
[pipeline_loop.py] (orchestrator)
       ↓
   ┌───┴────┬─────────┬──────────┬──────────────┐
   ↓        ↓         ↓          ↓              ↓
[MAFFT]  [MUSCLE]  [trimal]  [ESM-2]   [compute_extended_metrics]
   ↓        ↓         ↓          ↓              ↓
   └───┬────┴─────────┴──────────┘              ↓
       ↓                                        ↓
[ai_controller.py] ←── [RAG context]            ↓
       ↓                                        ↓
[LLM via Ollama]                               ↓
       ↓                                        ↓
   Decision ←───── Convergence Tracker ←── Hill Numbers
       ↓
   Updates params
       ↓
[rag_store] saves to knowledge base
```

### Key Components

1. **Alignment Runners** (msa_runner, muscle_runner, trimal_runner, esm_aligner)
2. **Metrics Engine** (extended_metrics with Hill numbers)
3. **Convergence Tracker** (convergence)
4. **AI Decision Controller** (ai_controller)
5. **RAG Layer** (rag_store, rag_retrieve, rag_context, rag_advanced, embeddings)
6. **Protein Embeddings** (protein_embeddings — ESM-2 650M)
7. **Knowledge Base** (TOON files + sparse NumPy index)
8. **Visualization & Dashboard** (visualization, dashboard.py)
9. **Export/Import** (export_import)
10. **Feedback System** (feedback_system)
11. **Batch Builder** (build_knowledge.py — safe, signal-handled, timeout-enforced)

---

## 3. Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.12 |
| Alignment | MAFFT v7.526, MUSCLE v5.3, trimal v1.5, ESM-2 (embedding-based) |
| LLM | Ollama (local) with llama3 model |
| LLM API | Custom `llm_core.py` (Ollama HTTP) |
| Package Manager | Conda (miniforge3) |
| Environments | `AI` (numpy/scipy/torch/transformers), `seqTools` (mafft/muscle/trimal) |
| Data Storage | TOON files + JSON configs + TF-IDF sparse NumPy index |
| Dependencies | numpy, scipy, torch, transformers, requests |

---

## 4. Folder Structure

```
/home/user/AI/
├── run_pipeline.py              # Entry point (CLI) with --warm-start-threshold
├── HANDOFF.md                   # This document
├── README.md                    # Project readme
├── FlowChart.md                 # Full Mermaid flowcharts (loop, decision engine, ESM, RAG, batch)
├── ALIGNMENT_SCORING.md         # Good/bad scoring reference (Hill, Shannon, thresholds)
├── ContactPairMapping.md        # Contact-pair mapping ideas (MI/attention, trim protection)
├── test.fasta                   # Test file (5 similar seqs)
├── test_variable.fasta          # Test file (8 diverse seqs, high gap)
├── setup.sh                     # Linux/Mac setup script
├── setup.bat                    # Windows setup script
├── scripts/
│   ├── __init__.py
│   ├── config.py                # Global config constants (CONDA_ENV, PROJECT_ROOT, paths)
│   ├── state.py                 # PipelineState container
│   ├── logger.py                # JSONL experiment logger
│   ├── msa_runner.py            # MAFFT wrapper (conda, --maxiterate clamped ≥1000)
│   ├── mafft_config.py          # MAFFT strategies/flags metadata
│   ├── muscle_runner.py         # MUSCLE v5 wrapper (-maxiters ≥1000, extra_args whitelist)
│   ├── trimal_runner.py         # trimal wrapper
│   ├── metrics.py               # Basic metrics
│   ├── extended_metrics.py      # sp_score, cs_score, conservation, gap_distribution, identity, Hill numbers
│   ├── convergence.py           # ConvergenceTracker
│   ├── ai_controller.py         # LLM decision logic + per-tool extra_args sanitization
│   ├── llm_core.py              # Ollama API wrapper (OLLAMA_TIMEOUT=120)
│   ├── embeddings.py            # TF-IDF (6-mers) + cosine similarity + fingerprint
│   ├── tfidf_sparse.py          # scipy.sparse TF-IDF index (build, query, save, load)
│   ├── toon_format.py           # TOON encoder/decoder (~670 lines, pure stdlib)
│   ├── protein_embeddings.py    # ESM-2 650M embedding + residue probability computation
│   ├── esm_aligner.py           # ESM-2 alignment (NW + embedding scoring)
│   ├── esm_analysis.py          # ESM conservation comparison, gap refinement, column report
│   ├── rag_store.py             # Save experiments (gzip append via subprocess)
│   ├── rag_retrieve.py          # Retrieve experiments (gzip -dc subprocess read)
│   ├── rag_context.py           # Build RAG context for LLM
│   ├── rag_advanced.py          # Confidence weighting, time decay, cross-validation
│   ├── knowledge_base.py        # MAFFT/trimal documentation
│   ├── feedback_system.py       # User rating (1-5 stars)
│   ├── export_import.py         # Knowledge base sharing (ZIP)
│   ├── visualization.py         # Dashboard, timeline, strategy stats
│   ├── format_tools.py          # Storage inspection commands
│   ├── clean_kb.py              # Knowledge base cleanup (dedupe/compress/backups)
│   ├── dashboard.py             # CLI dashboard tool
│   ├── pipeline_loop.py         # Main orchestrator (run_pipeline) + trim strategy picker
│   ├── build_knowledge.py       # Batch KB builder (safe, gzip resume sidecar)
│   ├── pfamSep.py               # Pfam FASTA splitter
│   ├── test_pipeline.py         # Test suite (~30 tests)
│   ├── run_from_file.py         # Test runner for specific FASTA
│   └── test_decision.py         # Pre-existing
├── envs/
│   ├── seqTools.yml             # Conda env for alignment tools (mafft, muscle, trimal)
│   └── ai.yml                   # Conda env for Python deps (numpy, scipy, torch, transformers)
├── knowledge/                   # Knowledge base (all data stored as .gz via gzip subprocess)
│   ├── experiments.jsonl.gz     # All experiment records (append-only gzip members)
│   ├── strategy_stats.toon.gz   # Per-strategy success rates (TOON tabular, gzipped)
│   ├── processed_families.txt.gz # Batch builder resume sidecar
│   ├── tfidf_index.npz          # scipy sparse CSR matrix
│   ├── tfidf_index_vocab.json   # term→index mapping + experiment_ids
│   ├── user_feedback.json       # User ratings
│   └── external_knowledge.json  # Custom knowledge entries
├── data/
│   ├── pfam/
│   │   ├── families/            # 30,007 family .faa files (11GB)
│   │   └── sample/              # 50 random families for testing
│   ├── Pfam-A.fasta             # Original Pfam dataset (13GB, archived)
│   └── README.txt               # Data placement instructions
├── models/
│   ├── models--facebook--esm2_t33_650M_UR50D/  # ESM-2 650M model (2.5GB)
│   └── README.txt               # Model placement instructions
├── logs/                        # Build logs and experiment logs
│   ├── batch_knowledge.log      # Batch builder log
│   └── <exp_id>.jsonl           # Per-experiment logs
├── trash/                       # Deprecated scripts
│   ├── process_uniref.py
│   ├── check_uniref_download.sh
│   ├── run_from_file.py
│   ├── test_decision.py
│   ├── test_ollama.py
│   ├── config.py
│   ├── state.py
│   └── HANDOFF_FINAL.md
├── WORKSTATION_REQUIREMENTS.md  # Hardware specs (4 tiers)
├── WORKSTATION_CONFIGURATIONS.tsv  # 15 GPU configs comparison
├── FutureUpdates.md             # Implementation roadmap
└── .gitignore                   # Excludes data/, models/, __pycache__/, *.fasta
```

---

## 5. Files Created/Modified

### Created (all in `/home/user/AI/scripts/`)

| File | Purpose |
|------|---------|
| msa_runner.py | MAFFT via conda |
| muscle_runner.py | MUSCLE v5 via conda (with ensemble, diversified, stratified, maxcc) |
| trimal_runner.py | trimal via conda |
| metrics.py | Basic metrics (gap_fraction, length) |
| extended_metrics.py | sp_score, cs_score, conservation, gap_distribution, identity, Hill numbers |
| convergence.py | ConvergenceTracker class |
| ai_controller.py | LLM decision controller (supports MAFFT/MUSCLE/ESM-2) |
| llm_core.py | Ollama API (pre-existing, slightly modified) |
| embeddings.py | TF-IDF (6-mers) + cosine similarity + fingerprint |
| protein_embeddings.py | ESM-2 650M embedding computation |
| esm_aligner.py | ESM-2 alignment (NW with embedding scoring) |
| rag_store.py | Store experiments (with TF-IDF, fingerprint, duplicate detection) |
| rag_retrieve.py | Retrieve relevant experiments (sparse search + warm start) |
| rag_context.py | Build RAG context string |
| rag_advanced.py | Confidence weighting, time decay, cross-validation |
| knowledge_base.py | MAFFT/trimal docs for context |
| feedback_system.py | User ratings + strategy weight adjustment |
| export_import.py | ZIP-based knowledge sharing |
| visualization.py | Dashboard, timeline, strategy performance |
| dashboard.py | CLI tool for all commands |
| pipeline_loop.py | Main orchestrator |
| build_knowledge.py | Batch KB builder (safe, signal handling, timeout enforcement) |
| pfamSep.py | Pfam FASTA splitter |
| test_pipeline.py | Test suite |
| logger.py | JSONL logger |
| config.py | Constants |
| state.py | State container |
| mafft_config.py | MAFFT metadata |

### Modified

- `run_pipeline.py` (root) - Now accepts FASTA file as CLI arg, with argparse

---

## 6. Completed Work

### Core Pipeline
- MAFFT integration (--auto, --localpair, --globalpair, --retree, --fast, --genafpair)
- MUSCLE v5 integration (align, progressive, super5, diversified, stratified, maxcc, perm, perturb, replicates)
- trimal integration (gentle only: `-gappyout`, `-automated1` — validated against trimAl v1.5; aggressive modes banned to keep indels)
- ESM-2 integration (embedding-based alignment via Needleman-Wunsch + star strategy)
- Iterative loop with convergence tracking
- Auto-trim fallback when MAFFT can't improve (overrides accept if gap > 0.15)
- Auto-switch cycles through 3 tools: mafft ↔ muscle ↔ esm2
- CLI entry point: `python run_pipeline.py <fasta> -n 5`

### Metrics
- Basic: gap_fraction, alignment_length, avg_sequence_length
- Extended: sp_score, cs_score, conservation, gap_distribution, identity
- Hill numbers: hill_q0_avg (richness), hill_q1_avg (true diversity), hill_q2_avg (inverse Simpson)
- Column report: problematic columns with high diversity or many gaps
- All metrics in `extended_metrics.compute_extended_metrics()`

### AI Controller
- Knows about MAFFT, MUSCLE, trimal, ESM-2
- Actions: accept, refine, trim, change_tool, change_strategy, stop
- LLM prompt includes Hill number metrics + column report
- ESM-2 decision rules: "hill_q1_avg > 10 → consider esm2", "identity < 0.2 AND gap > 0.25 → use esm2"
- Validates and retries on bad JSON
- Safety override: refuses accept if gap > 0.15

### RAG Layer
- Stores experiments with TF-IDF embeddings
- Stores sequence fingerprints (nucleotide/protein)
- Sparse TF-IDF cosine search via scipy (replaces Jaccard)
- Warm start: detects similar experiments and passes action history to LLM
- Duplicate detection (>70% similarity skips)
- Strategy stats (success/failure rates)
- Failure lessons (what NOT to do)
- External knowledge (MAFFT/trimal docs)
- Confidence weighting (high/medium/low)
- Time decay (older experiments weighted less)
- Cross-validation (test accuracy)
- Feedback integration (user ratings adjust weights)
- TOON format for LLM-readable storage
- Auto-fallback readers (TOON → JSONL/JSON)
- **ESM gap-band learning (NEW)**: `actions_tried[].tool` recorded; probe rule tries esm2 once when gap ∈ [0.12, 0.40] (unlimited seqs); `esm_gap_stats.toon.gz` tracks per-band n/success/avg Δgap; learned threshold = lowest band with **lo ≥ 0.10**, n≥3, ≥50% success (else 0.25) — bands lo&lt;0.10 ignored so thr never becomes 0; drives `_rule_based_decision` + one LLM hint; dashboard `ESM-2 BY GAP BAND`
- **ESM post-correction + warm-start seeding (NEW)**: `esm_post=if_bad` everywhere (build + CLI default) — refine only when gap>0.12 and band advice≠AVOID (empty stats→explore so bad gaps still teach; never on good gaps — too slow otherwise); `esm_post_stats.toon.gz` + `get_esm_post_advice`; records store `esm_post_applied`/`esm_gap_before`/`esm_gap_after`; warm match seeds `current_params` from `derive_warm_plan` (start tool/strategy, avoid trims, direct-esm flag) and LLM context includes per-action `tool`

### Batch Builder (NEW)
- Safe signal handling (SIGINT/SIGTERM for graceful shutdown)
- Timeout enforcement (kills stuck pipelines after timeout)
- Child process cleanup (kills mafft/muscle/trimal subprocesses)
- Resume support via gzip sidecar (`knowledge/processed_families.txt.gz`, one gzip member per family)
- Min sequences filter (skips families with < N sequences)
- Dry-run mode (preview families without processing)
- Progress logging with ETA

### Gzip Knowledge Storage (NEW)
- All experiment data stored as `experiments.jsonl.gz` — append-only, one gzip member per record
- Writes: `subprocess.run(["gzip", "-c"], input=..., stdout=file)` piped from Python
- Reads: `subprocess.run(["gzip", "-dc", path])` stdout piped into Python parser
- Same pattern for `strategy_stats.toon.gz` and `processed_families.txt.gz`
- Single canonical store (no plain/TOON shadowing; stale-gz priority bug fixed)

### ESM Post-Alignment Analysis (NEW)
- `esm_analysis.py`: per-residue probability distributions via ESM-2 MLM head
- Compares ESM conservation vs observed Hill q1 per column (Pearson correlation)
- Misaligned column detection (ESM conserved but alignment diverse)
- Optional gap refinement with position-specific penalties: `gap_open * (1 + alpha * conservation)`
- CLI: `--esm-analysis`, `--esm-refine-gaps`, `--esm-alpha` (default 2.0)
- Human-readable column report including last column

### Decision Hardening (NEW)
- Rule-based decision layer (`_rule_based_decision`) skips LLM for obvious cases
- Metrics echo validation (`_check_metrics_echo`) catches LLM hallucinated metrics
- Per-tool `extra_args` sanitization (`_sanitize_extra_args`) — MAFFT flags can never reach muscle
- MUSCLE `extra_args` whitelisted against real v5 flags; runner double-filters as defense-in-depth
- `change_tool` clears `extra_args` (no cross-tool flag leakage)
- Trimal **gentle-only** rotation: `_pick_trim_strategy` cycles `-gappyout` → `-automated1` only; `-strict`/`-strictplus`/`-nogaps` rejected everywhere (rule, LLM sanitize, runner) — they delete gap/indel columns
- MUSCLE `-maxiters` clamped to ≥1000 (passed after input file; muscle 5.3 may report "not used")
- MAFFT `--maxiterate` clamped to ≥1000 in `msa_runner.py`

### Pfam Data Processing
- Split Pfam-A.fasta (13GB, 63.8M seqs) into 30,007 family .faa files
- Large families capped at 8000 sequences during processing
- Sorted by size ascending for efficient processing

### Tooling
- Dashboard (`python dashboard.py --dashboard`)
- Strategy performance (`--strategies`)
- Learning timeline (`--timeline`)
- Cross-validation (`--validate`)
- Report generation (`--report`)
- Export/import knowledge (`--export`, `--import`)
- Feedback (`--feedback <exp_id>`)
- Clean old experiments (`--clean 10`)
- Format info (`--format-info`)
- Migrate to TOON (`--migrate-to-toon`)
- Sparse index info (`--sparse-info`)
- LLM context preview (`--toon-context`)
- Compare formats (`--compare-formats`)
- Inspect experiments (`--inspect [id]`)

---

## 7. Pending Tasks

### High Priority
- [ ] Run batch builder on all 30,007 Pfam families
- [ ] Test warm start effectiveness as KB grows
- [ ] Test pipeline on novel protein families

### Medium Priority
- [ ] Add web dashboard (Flask/Streamlit)
- [ ] Add phylogenetic tree integration
- [ ] Add parallel iteration (try multiple strategies simultaneously)
- [ ] Improve ESM-2 alignment quality (affine gap penalties, larger models)

### Done
- [x] Migrate storage to TOON + sparse NumPy
- [x] Dashboard commands for storage inspection
- [x] Warm start from similar experiments
- [x] User-defined warm start threshold
- [x] IDF document frequency fix (sparse search works)
- [x] Strategy key quote stripping
- [x] Ollama timeout fix
- [x] Experiment inspect command
- [x] Hill numbers for column diversity scoring
- [x] ESM-2 embedding-based alignment (650M model)
- [x] 3-tool auto-switch (mafft ↔ muscle ↔ esm2)
- [x] Hill numbers in LLM prompt + column report
- [x] Safe batch builder with signal handling
- [x] Timeout enforcement for stuck pipelines
- [x] Min sequences filter for small families
- [x] Pfam data processing (30,007 families)
- [x] Resume support for batch building (gzip sidecar)
- [x] Gzip knowledge storage via gzip subprocess (append + read)
- [x] ESM probability/conservation post-analysis (`--esm-analysis`)
- [x] Rule-based decision layer + metrics echo validation
- [x] Per-tool extra_args sanitization (fixes MUSCLE invalid-command crash)
- [x] Trimal strategy rotation (gentle-only: `-gappyout` ↔ `-automated1`, indels kept)
- [x] MUSCLE -maxiters ≥1000 + MAFFT --maxiterate ≥1000 clamps
- [x] KB cleanup (4.3GB backups removed, stale gz shadowing fixed)
- [x] TOON decoder quote-accumulation bug fixed

### Low Priority
- [ ] Docker containerization
- [ ] Cloud deployment
- [ ] Multi-user support
- [ ] Authentication
- [ ] API endpoint (FastAPI)

---

## 8. Bugs / Issues / Failed Attempts

### Resolved
1. **MAFFT not in PATH** → Fixed by using `conda run -n seqTools mafft` instead of `which mafft`
2. **Trimal version flag wrong** → Changed from `-version` to `--version`
3. **Indentation error in trimal_runner** → User fixed manually
4. **Extra args carrying trimal options to MAFFT** → Added `just_trimmed` flag to skip extra_args update
5. **Log showing empty extra_args** → Reordered logging to happen AFTER params update
6. **Path issues in dashboard** → Fixed with `pathlib.Path(__file__).parent.resolve()`
7. **Sequence signature type mismatch** (set vs list) → Converted list to set on retrieval
8. **MUSCLE v5 argument order** → Moved `-perm`/`-perturb`/`-replicates` AFTER input file
9. **diversified doesn't allow -perm** → Expected behavior (use perm with regular align)
10. **compute_metrics not defined** → Changed to `compute_extended_metrics`
11. **IDF formula bug** → Used term count instead of document frequency; sparse search returned no results
12. **Strategy key quote accumulation** → Keys gained `\"` on every round-trip
13. **Ollama timeout** → Default 30s too short for large prompts; set to 120s
14. **TOON decoder escape handling** → `\"` un-synced quote tracking in `_find_top_colon`
15. **Zombie processes on Ctrl+Z** → Fixed with signal handling and SIGTSTP ignore
16. **Child processes not cleaned up** → Fixed with process group management
17. **MUSCLE "Invalid command line" crash (PF19758)** → LLM echoed MAFFT default flags (`--maxiterate 0 --op 1.53 --ep 0.0 --nuc`) into `extra_args` for a muscle run; fixed with per-tool `_sanitize_extra_args` in `ai_controller.py`, extra_args clearing on `change_tool`, and a muscle_runner whitelist
18. **Stale `experiments.jsonl.gz` shadowed live JSONL** → `load_knowledge` preferred a stale .gz (37 records) over the live file (39); fixed by making gzip the single canonical store
19. **Trimal always used `-strict`** → strategy hardcoded in rule engine, converged override, and LLM prompt; fixed with `_pick_trim_strategy()` severity + rotation
20. **MUSCLE strategy/algorithm key mismatch** → pipeline stored mode in `params["strategy"]` but runner read `params["algorithm"]` (super5/progressive silently ignored); runner now reads both
21. **`--maxiters` before input file crashes muscle 5.3** → must be placed after the input file; `-progressive` rejects it entirely (skipped for that algorithm)
22. **Batch builder resume never skipped anything** → compared experiment UUIDs against family filenames; fixed with gzip sidecar `processed_families.txt.gz`
23. **TOON decoder quote accumulation** → numeric string keys re-quoted exponentially on round-trip (`_decode_key()` added)
24. **TRIMAL_STRATEGIES contained flags that don't exist in trimAl v1.5** → `-gapped`, `-nogap`, `-soft`, `-verystrict` are invalid (only `-strict` worked — the real reason trimal "never changed strategy"); replaced with validated flags: `-gappyout`, `-strict`, `-strictplus`, `-automated1`, `-nogaps`

### Known Issues
1. **LLM doesn't always follow rules** (e.g., says "gap < 0.1" when it's 0.22) - mitigated with safety override + metrics echo validation
2. **AI rarely switches to MUSCLE** - prompt should encourage it more
3. **Duplicate detection uses k-mer Jaccard for `_is_duplicate` but TF-IDF for `find_similar_experiments`** - inconsistent
4. **MUSCLE 5.3 accepts `-maxiters` but reports "Option -maxiters not used"** - muscle 5 has no iteration cap; flag is passed for compatibility (min 1000), real quality knobs are `-replicates`/`-perm`
5. **`-progressive` algorithm rejects `-maxiters`** - flag skipped for that algorithm only

### Failed Attempts
1. **MUSCLE v3 syntax** - Tried using `muscle -in` (v3 syntax) but user has v5 which uses different CLI
2. **stratified with -perm** - MUSCLE refuses this combination
3. **UniRef approach** - Abandoned in favor of Pfam for knowledge base building

---

## 9. Key Decisions and Reasoning

| Decision | Reasoning |
|----------|-----------|
| **TOON + sparse storage** | LLM-readable text + fast cosine search via numpy/scipy |
| **Two conda envs** | `AI` for numpy/scipy/torch, `seqTools` for alignment tools - avoids conflicts |
| **TF-IDF over Word2Vec** | No external dependencies, better than k-mer Jaccard |
| **Ollama local LLM** | User has it installed, no API costs |
| **Conda env `seqTools`** | User specified it explicitly |
| **Safety override on accept** | LLM often hallucinates good metrics; force retry if bad |
| **Auto-trim on converged-but-bad** | If alignment is stable but poor, trim rather than retry MAFFT |
| **Warm start from similar experiments** | Avoids repeating failed strategies; LLM gets full action history |
| **User-defined warm start threshold** | Flexibility to be aggressive (0.6) or conservative (0.9) |
| **MUSCLE v5 syntax** | Updated to support diversified/stratified/maxcc ensemble features |
| **Hill numbers for column scoring** | Multi-scale diversity: q0=richness, q1=true diversity, q2=inverse Simpson |
| **ESM-2 for protein alignment** | Captures deep evolutionary signal beyond k-mer matching |
| **ESM-2 650M over ProtT5** | Lighter (1.5GB vs 6GB), still effective for alignment |
| **Star alignment for ESM-2** | Center-star strategy for multiple sequences with embedding scoring |
| **3-tool auto-switch** | mafft ↔ muscle ↔ esm2 cycle for maximum coverage |
| **Pfam over UniRef** | 30,007 well-curated families vs single large dataset |
| **Safe batch builder** | Signal handling, timeout enforcement, child cleanup for long runs |

---

## 10. Coding Conventions

- **Type hints** on all function signatures
- **No comments** unless explicitly asked
- **Pure stdlib** (no pip install required)
- **Path safety**: Use `pathlib.Path(__file__).parent.resolve()` for absolute paths
- **Error handling**: Raise `RuntimeError` with clear message on tool failure
- **Subprocess**: Use `conda run -n seqTools <tool>` pattern
- **Logging**: Use existing `logger.log_step()` then `logger.save(exp_id)`
- **JSON storage**: One record per line, easy to append

---

## 11. Dependencies

### System Requirements
- Linux
- Conda (miniforge3)
- Python 3.12
- Two conda envs:
  - `AI` env: numpy, scipy, torch, transformers (for RAG/ML)
  - `seqTools` env: mafft, muscle v5, trimal (for alignment tools)

### Python
- `numpy` + `scipy` (sparse matrices, TF-IDF index)
- `torch` + `transformers` (ESM-2 protein embeddings)
- `requests` (Ollama HTTP calls)
- Standard library: `subprocess`, `tempfile`, `shutil`, `os`, `json`, `uuid`, `pathlib`, `datetime`, `math`, `re`, `collections`, `multiprocessing`, `signal`
- `ollama` server running locally (default: `http://localhost:11434`)

---

## 12. Environment Setup

```bash
# 1. Create conda env (if not exists)
conda create -n seqTools python=3.12

# 2. Install tools
conda install -n seqTools -c bioconda mafft
conda install -n seqTools -c bioconda muscle
conda install -n seqTools -c bioconda trimal

# 3. Install Ollama (system-level)
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &
ollama pull llama3

# 4. Verify
conda run -n seqTools mafft --version
conda run -n seqTools muscle --version
conda run -n seqTools trimal --version
```

---

## 13. Commands Reference

### Run Pipeline
```bash
cd /home/user/AI
python run_pipeline.py <fasta_file> -n <iterations> -t <tolerance> -p <patience> -w <warm_start_threshold>
# Example:
python run_pipeline.py test.fasta -n 10
python run_pipeline.py test.fasta -n 5 -w 0.9   # conservative warm start
python run_pipeline.py test.fasta -w 0.6          # aggressive warm start
python run_pipeline.py test.fasta --esm-analysis  # + ESM conservation analysis
python run_pipeline.py test.fasta --esm-analysis --esm-refine-gaps --esm-alpha 2.0
```

### Build Knowledge Base
```bash
cd /home/user/AI

# Preview families
python scripts/build_knowledge.py --dry-run --limit 100

# Run with safety features
python scripts/build_knowledge.py --limit 100 --timeout 300 --min-seqs 10

# Resume after interrupt
python scripts/build_knowledge.py --resume --limit 100

# Run all families (background) — preferred launcher
./build.sh                 # fresh wipe + full batch
./build.sh --resume        # continue existing KB
# PID: logs/batch_knowledge.pid (also build.pid) · log: logs/batch_knowledge.log
```

### Dashboard
```bash
cd /home/user/AI/scripts
python dashboard.py --dashboard
python dashboard.py --strategies
python dashboard.py --timeline
python dashboard.py --validate
python dashboard.py --report
python dashboard.py --export my_export.zip
python dashboard.py --import my_export.zip
python dashboard.py --clean 10
python dashboard.py --feedback <exp_id>
```

### Tests
```bash
cd /home/user/AI/scripts
python test_pipeline.py
```

### Direct Tool Use
```python
from msa_runner import run_msa
from muscle_runner import run_muscle
from trimal_runner import run_trimal
from extended_metrics import compute_extended_metrics
from ai_controller import decide_action
```

---

## 14. Important Configs

### `run_pipeline.py` defaults
- `max_iterations = 10`
- `convergence_tolerance = 0.01`
- `convergence_patience = 2`
- `warm_start_threshold = 0.8`
- `esm_analysis = False` (`--esm-analysis` enables post-alignment ESM conservation analysis)
- `esm_refine_gaps = False` (`--esm-refine-gaps` enables ESM-informed gap refinement)
- `esm_alpha = 2.0` (`--esm-alpha` scales position-specific gap penalties)

### Pipeline behavior
- Default tool: `mafft`
- Default strategy: `--auto`
- Auto-trim threshold: `gap_fraction > 0.15`
- Accept threshold: `gap_fraction < 0.1 AND instability < 0.01`
- Trimal strategy picker: always gentle — start `-gappyout`, rotate to `-automated1` on repeated trims; `-strict`/`-strictplus`/`-nogaps` rejected (indels must survive)
- MUSCLE: `-maxiters` always passed (min 1000, after input file); MAFFT `--maxiterate` clamped to min 1000

### AI Controller
- `allowed_actions`: accept, refine, trim, change_tool, change_strategy, stop
- `allowed_tools`: mafft, muscle, esm2
- `allowed_modes`: auto, localpair, globalpair, fast, align, progressive, super5, local, global
- Retries LLM once on bad JSON

### RAG
- Duplicate threshold: 70% sequence similarity
- TF-IDF k-mer size: 6
- Warm start threshold: 80% cosine similarity (configurable via `-w`)
- Retrieval: top 3 experiments
- Sparse cosine search via scipy CSR matrix

### Batch Builder
- Timeout per family: 300s (configurable via `--timeout`)
- Min sequences: 10 (configurable via `--min-seqs`)
- Max sequences per family: 8000 (configurable via `--max-seqs`)
- Log file: `logs/batch_knowledge.log`

---

## 15. APIs / Models / Tools Used

| Tool | Version | API |
|------|---------|-----|
| MAFFT | 7.526 | CLI via `conda run` |
| MUSCLE | 5.3 | CLI via `conda run` |
| trimal | 1.5.rev1 | CLI via `conda run` |
| Ollama | latest | HTTP POST `http://localhost:11434/api/generate` |
| llama3 | latest | via Ollama |
| ESM-2 | 650M | Local model (HuggingFace transformers) |

---

## 16. Current Blockers

**Ollama stability under batch load** — llama3 wedges when the batch sends large RAG prompts (generate hangs, all subsequent requests time out). Mitigations in place: `num_predict=200` cap, LLM-failure fallback in pipeline, restart via `sudo systemctl restart ollama`. Watch `logs/batch_knowledge.log` for consecutive `Read timed out` — if seen, restart Ollama and `--resume` the batch.

Other observations:
- AI tends to use `mafft` for everything; doesn't switch to MUSCLE (rule layer helps)
- AI hallucinates about metrics - mitigated by safety override + metrics echo check
- Knowledge base building takes ~2-3 minutes per family (LLM-bound when it works)

---

## 17. Next Recommended Steps

1. **Run fresh batch build** — `./build.sh` (wipes knowledge/+logs/, starts full 30k Pfam run, PID in `logs/batch_knowledge.pid` + `build.pid`); or `./build.sh --resume` after interrupt
2. **Monitor batch health** — `tail -f logs/batch_knowledge.log`; consecutive `Read timed out` = Ollama wedged → `sudo systemctl restart ollama` → `./build.sh --resume`
3. **If Ollama keeps wedging**: reduce prompt size (cap `rag_context` in `ai_controller._build_prompt`), or switch batch to a smaller model, or add Ollama keep-alive/queue tuning
4. **Test warm start effectiveness** — Verify that similar experiments are found and used once KB has volume
5. **Test on novel proteins** — Run pipeline on unseen protein families
6. **Muscle strategy diversity** — muscle always ran `align`; consider enabling `super5`/`progressive` for large/diverse families
7. **Watch indels** — if finals still hit gap=0 with no trim, inspect ESM star-aligner (gentle trims alone may not be the only path)
8. **Add web UI** — Streamlit or Flask for browser-based interaction
9. **Multi-objective optimization** — Currently only optimizes gap_fraction
10. **Improve ESM-2 alignment** — Affine gap penalties, larger models (ESM-2 15B)

### Future direction (see also FutureUpdates.md)
- **Phase 2**: pLM perplexity-based difficulty prediction → pre-select tools before running
- **Phase 3**: ProtT5/embedding-based alignment scoring as extra metric
- **Phase 4**: Hybrid tool selection from embedding distances
- **Knowledge quality**: use accumulated `strategy_stats` to bias RAG recommendations; retire strategies with high failure rates
- **Batch throughput**: parallelize families (careful with Ollama queue — `OLLAMA_NUM_PARALLEL=1` currently)

---

## 18. Continuation Plan (for new session)

### Step 1: Verify environment
```bash
conda run -n seqTools mafft --version
conda run -n seqTools muscle --version
conda run -n seqTools trimal --version
ollama list
ls /home/user/AI/scripts/
ls /home/user/AI/knowledge/
ls /home/user/AI/data/pfam/families/ | wc -l  # Should be 30007
```

### Step 2: Run smoke test
```bash
cd /home/user/AI
python run_pipeline.py test.fasta -n 2
python scripts/build_knowledge.py --dry-run --limit 10
```

### Step 3: If something is broken
- **MAFFT error**: Check conda env, reinstall with `conda install -n seqTools -c bioconda mafft`
- **MUSCLE error**: Check v5 syntax in `muscle_runner.py` (MUSCLE v3 uses different flags)
- **LLM error**: `ollama serve` then `ollama pull llama3`
- **Import error**: Check `sys.path.insert` in `run_pipeline.py` adds `scripts/`
- **Batch builder hangs**: Check `logs/batch_knowledge.log` for TIMEOUT entries

### Step 4: Continue development
- Read this handoff doc first
- Check pending tasks in section 7
- The code is in `/home/user/AI/scripts/`
- The knowledge base is in `/home/user/AI/knowledge/`
- Logs are in `/home/user/AI/logs/`
- Pfam families are in `/home/user/AI/data/pfam/families/`

### Key file to start with
- `pipeline_loop.py` - Main orchestrator
- `ai_controller.py` - LLM decision logic
- `extended_metrics.py` - 5 new metrics
- `embeddings.py` - TF-IDF + fingerprint
- `muscle_runner.py` - MUSCLE v5 with ensemble support
- `build_knowledge.py` - Safe batch builder

### Important: Always test after changes
```bash
cd /home/user/AI
python run_pipeline.py test.fasta -n 2
```

---

## 19. Session Statistics

- **Duration:** ~10 weeks (June 2 - September 5, 2026), extended to October 30, 2026
- **Files created:** 40+ Python files
- **Lines of code:** ~5000+ LOC
- **Iterations tested:** 60+ pipeline runs
- **Knowledge base:** reset clean (gzip store via subprocess, backups removed)
- **Tools integrated:** 4 (MAFFT, MUSCLE, trimal, ESM-2)
- **Metrics:** 14 (3 basic + 5 extended + 6 Hill numbers)
- **RAG features:** 8 (storage, retrieval, TF-IDF, sparse search, warm start, duplicate detection, weighted recs, cross-validation)
- **Dashboard commands:** 14
- **Test suite:** ~30 tests passing
- **Pfam families:** 30,007 (ready for batch processing)
- **Batch builder:** Safe with signal handling, timeout enforcement, gzip resume sidecar
- **Documentation:** README.md, HANDOFF.md, FutureUpdates.md, WORKSTATION_REQUIREMENTS.md, WORKSTATION_CONFIGURATIONS.tsv, FlowChart.md (Mermaid system map), ALIGNMENT_SCORING.md (scoring reference), ContactPairMapping.md (contact-pair ideas)

---

## 20. Contact / Context for Continuation

**User:** Inder (AI bioinformatics researcher)
**Goal:** Build AI that runs MSA and evaluates if good/bad
**Key insight:** System learns over time which strategies work for which problems
**Demo audience:** Someone who needs to see the AI learn and improve alignment

When continuing, ask user:
1. What's the most important next feature?
2. Are there specific test cases to validate against?
3. Any new tools to integrate (ClustalW, ProbCons, etc.)?
4. What evaluation criteria matter most (speed, accuracy, both)?

---

## Update Log

### June 2, 2026: Storage migrated to TOON + Sparse NumPy
- New files: `toon_format.py`, `tfidf_sparse.py`
- Modified: `rag_store.py`, `rag_retrieve.py`, `rag_advanced.py`
- TOON decoder bugs fixed (4 issues)

### June 2, 2026 (later): TOON/sparse storage validated end-to-end
- Pipeline run time: ~2.5 minutes for test_variable.fasta
- Fixes applied: path resolution, escape handling, tabular array detection

### June 2, 2026 (final): Dashboard + Demo + Tests Complete
- New file: `format_tools.py` (5 commands)
- Wired into `dashboard.py`
- 5 new tests added (10/10 passing)

### June 9, 2026: Warm Start Feature
- New function: `find_warm_start()` in `rag_retrieve.py`
- CLI arg: `-w` for warm start threshold
- New test: `test_warm_start()` (12/12 passing)

### June 10, 2026: Hill Numbers + ESM-2 Alignment
- New metrics: Hill numbers (q0, q1, q2)
- New modules: `protein_embeddings.py`, `esm_aligner.py`
- Pipeline integration: ESM-2 as third tool
- New tests: `test_hill_numbers()`, `test_esm_alignment()` (15/15 passing)

### September 5, 2026: Safe Batch Builder + Pfam Data
- New module: `build_knowledge.py` (safe, signal handling, timeout enforcement)
- Pfam data processed: 30,007 family .faa files
- Features: resume support, min sequences filter, dry-run mode
- Safety: signal handlers, child process cleanup, timeout enforcement
- Documentation: README.md, HANDOFF.md, FutureUpdates.md updated

### September 5, 2026: Path-Independent Scripts + Portable Drive
- Created `scripts/config.py` with shared constants (CONDA_ENV, PROJECT_ROOT, paths)
- Updated all runners to import from config (removed duplicate CONDA_ENV)
- Fixed hardcoded paths in `pfamSep.py` and `build_knowledge.py`
- Created `envs/seqTools.yml` and `envs/ai.yml` for conda environment setup
- Created `setup.sh` (Linux/Mac) and `setup.bat` (Windows) setup scripts
- All scripts now work from any directory (tested from /tmp)
- Portable drive setup: code, families, models go on drive; knowledge base stays local

### October 29, 2026: Gzip Storage + ESM Probability Analysis
- Knowledge base migrated to gzip: `experiments.jsonl.gz` (append-only, one gzip member per record), `strategy_stats.toon.gz`
- All reads/writes via shell `gzip -c`/`gzip -dc` subprocess pipes in Python (`rag_store.py`, `rag_retrieve.py`)
- Removed `knowledge/backup/` (~4.3GB) and wiped stale knowledge files
- Fixed stale-gz priority bug (stale .gz shadowed live JSONL)
- New `esm_analysis.py`: ESM-2 MLM head residue probabilities, conservation vs Hill q1 correlation, misaligned column report, position-specific gap refinement
- New CLI flags: `--esm-analysis`, `--esm-refine-gaps`, `--esm-alpha`
- TOON decoder `_decode_key()` fix (quote accumulation)
- Rule-based decision layer + metrics echo validation + TF-IDF duplicate detection
- 4 new ESM tests (22 total)

### October 30, 2026: MUSCLE/TRIMAL Hardening + Docs
- Root-caused PF19758 crash: LLM sent MAFFT defaults as muscle extra_args → added `_sanitize_extra_args` per-tool whitelist in `ai_controller.py` + runner-level filter
- `change_tool` now clears `extra_args` (no cross-tool leakage)
- MUSCLE `-maxiters` min 1000 (after input file; skipped for `-progressive` which rejects it); MAFFT `--maxiterate` min 1000 clamp
- Fixed `strategy` vs `algorithm` key mismatch (muscle modes super5/progressive now reachable)
- Trimal strategy rotation via `_pick_trim_strategy()` (severity + rotation; removed 3 hardcoded `-strict` sites)
- Batch builder resume rewritten: gzip sidecar `processed_families.txt.gz` (old UUID-vs-filename comparison never matched)
- `build_knowledge.py`, `README.md`, `HANDOFF.md` updated; knowledge base wiped clean

### November 4, 2026: Trimal Flag Validation + LLM Safeguards + Batch Diagnosis
- **Trimal flags validated against installed trimAl v1.5.rev1**: old set (`-gapped`, `-nogap`, `-soft`, `-verystrict`) mostly invalid — only `-strict` worked (root cause of "trimal never changes strategy"). Replaced with `-gappyout`, `-strict`, `-strictplus`, `-automated1`, `-nogaps` across `trimal_runner.py`, `pipeline_loop.py`, `ai_controller.py`, `knowledge_base.py`, tests, docs. Bug #24 in HANDOFF.
- **Batch knowledge build launched** (PID file: `logs/batch_knowledge.log`, `logs/batch_knowledge.pid`); ~96 families processed before Ollama issues
- **Ollama llama3 repeatedly wedged** during batch: every `/api/generate` hung (read timeout 120s, server `500 | 2m0s`); families that fired rule-based accept without LLM still succeeded (`1 iters`). Fix: `sudo systemctl restart ollama`. Pattern: worked after restart, wedged once batch sent large RAG prompts
- **LLM safeguards added**:
  - `llm_core.py`: `options: {"num_predict": 200}` caps generation length (prevents runaway generations wedging the runner)
  - `pipeline_loop.py`: `decide_action` wrapped in try/except — on LLM failure falls back to safe decision (accept if gap<0.15, else severity-based trim) instead of failing the family
- **Verified**: 22/22 tests pass; strategy rotation confirmed in `knowledge/strategy_stats.toon.gz` (trim used `-gappyout`/`-strict`/`-strictplus`/`-nogaps`); tool usage from logs: mafft 487 iters (mostly `--auto`, 8 strategy switches), muscle 56 (always `align`), esm2 66 (`local`)
- **Batch status at session end**: not running; resume sidecar has 96 families; failed families not marked → `--resume` retries them

### November 6, 2026: Dashboard Split + Gap Stats + ESM Gap Probing + Post-Correction Learning
- **Stats split by problem type** (hard first, good_quality = reference): dashboard outcomes/Δgap block, tool usage, strategy performance, learning curve; `--validate` shows 2 samples per problem type
- **Mean/median gap reporting** replaced raw `avg Δgap`: `gap init/final mean+med` + `reduction mean/med %` (`(init−final)/init×100`, init=0 skipped); helper `_gap_set_stats` in `visualization.py`
- **Learning curve rewritten**: cumulative-by-n checkpoints (~4 quartiles per problem type) + Growth line first→last — no more date buckets
- **ESM gap-band learning (this change)**:
  - `actions_tried[].tool` stored (`rag_store.py` + `_tool_used` stamped in `pipeline_loop`)
  - Probe rule: gap ∈ [0.12, 0.40], esm2 not yet tried → `change_tool esm2` once (unlimited seqs); also fires when learned threshold exists and gap > thr
  - `esm_gap_stats.toon.gz`: per 0.05 band {n, success, sum_delta}; success = Δgap≥0.02 or after<0.15
  - `get_esm_threshold()`: lowest band n≥3 & ≥50% success else default 0.25; `get_esm_gap_hint()` injected into LLM rag_context
  - Dashboard: `ESM-2 BY GAP BAND` section (`--tools` / `--dashboard`)
- **ESM post-correction learning + warm-start seeding (this change)**:
  - `esm_post` mode: `always` (build_knowledge) | `if_bad` (CLI default, gap>0.12 and advice≠avoid) | `off`
  - After main loop: optional `run_esm_analysis(refine_gaps=True)`; store `esm_post_applied`, `esm_gap_before/after` on record
  - `esm_post_stats.toon.gz` + `get_esm_post_advice` (use/avoid/explore) + `get_esm_post_hint` in LLM context; dashboard `ESM POST-CORRECTION BY GAP BAND`
  - `derive_warm_plan`: start = lowest-gap successful align (may be esm2 = direct esm), plan = later success steps, avoid = failed; seeds `current_params` + failed trim flags into `trim_tried`; context lines include `tool=`
- **Tests**: 30/30 (new `test_esm_post_stats`, `test_esm_post_decision_gate`, `test_derive_warm_plan`)
- **Batch**: `esm_post` switched `always`→`if_bad` (selective post-correction; ~7% of finals would fire); relaunch with `--resume`

### November 11, 2026: Indel-Safe Trims + ESM Threshold Hygiene + Fresh Build
- **Gentle-only trimal** (indels must survive): `TRIM_STRATEGY_ORDER = ["-gappyout", "-automated1"]`; severity logic always starts `-gappyout`; rule paths + LLM fallback use `-gappyout`; `ai_controller` prompt/sanitize and `trimal_runner` reject `-strict`/`-strictplus`/`-nogaps` (they wipe gap columns — 26 KB records went init>0.05 → final 0)
- **ESM threshold no longer 0**: `get_esm_threshold()` ignores bands with `lo < 0.10` (the 0.00–0.05 band was 100% “success” via `after<0.15` and forced thr=0, so every gap fired ESM). Live thr now 0.10 (from `0.10–0.15`, n=18, 78%)
- **Resume sidecar hardened**: `get_processed_families` runs `gzip -dc` with `check=False` (rc 0|2) so multi-member trailing garbage no longer crashes `--resume` (78 families readable)
- **Tests updated**: gentle trim expectations, sanitize rejects aggressive flags, low-gap band ignored for threshold → **~30/30 pass**
- **`build.sh` launcher**: default wipes `knowledge/` + `logs/` then starts full batch with AI-env python; `--resume` keeps KB; PID → `logs/batch_knowledge.pid` + `build.pid`; log → `logs/batch_knowledge.log`
- **Docs**: README / HANDOFF / FutureUpdates updated for gentle trims, thr filter, `./build.sh`
- **KB wiped then rebuilt**: fresh `./build.sh` launched **2026-11-11 11:11** — PID `logs/batch_knowledge.pid` = **18777** (parent spawn 1843), log `logs/batch_knowledge.log`, `esm_post=if_bad`, gentle trims only
