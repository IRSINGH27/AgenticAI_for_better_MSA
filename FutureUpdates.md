# Future Updates - Protein Language Model Integration

**Created:** June 9, 2026
**Last Updated:** November 11, 2026
**Status:** Partially implemented

## Implemented

### Hill Numbers (June 10, 2026)
- Multi-scale column diversity scoring (q0, q1, q2)
- Column report for LLM context
- Added to `extended_metrics.py`

### ESM-2 Alignment (June 10, 2026)
- ESM-2 650M model for protein embeddings
- Needleman-Wunsch with embedding cosine similarity
- Star alignment for multiple sequences
- Added as third tool option (mafft ↔ muscle ↔ esm2)
- New files: `protein_embeddings.py`, `esm_aligner.py`

### Safe Batch Builder (September 5, 2026)
- Signal handling (SIGINT/SIGTERM for graceful shutdown)
- Timeout enforcement (kills stuck pipelines after timeout)
- Child process cleanup (kills mafft/muscle/trimal subprocesses)
- Resume support (skips already-processed families)
- Min sequences filter (skips families with < N sequences)
- Dry-run mode (preview families without processing)
- New file: `build_knowledge.py`

### Pfam Data Processing (September 5, 2026)
- Split Pfam-A.fasta (13GB, 63.8M seqs) into 30,007 family .faa files
- Large families capped at 8000 sequences during processing
- Sorted by size ascending for efficient processing
- New file: `pfamSep.py`

## Planned (Not Yet Implemented)

### Near-term (November 2026) — Knowledge Base Build
- **Fresh full build**: `./build.sh` — wipes `knowledge/` + `logs/`, starts batch with AI-env python, writes PID to `logs/batch_knowledge.pid` and `build.pid`
- **Resume after Ollama restart**: `./build.sh --resume` (keeps sidecar; failed families are not marked processed)
- **Monitor Ollama stability**: consecutive `Read timed out` in `logs/batch_knowledge.log` = runner wedged → `sudo systemctl restart ollama` → `./build.sh --resume`
- If prompt size is the wedge trigger: cap `rag_context` length in `ai_controller._build_prompt`
- Muscle strategy diversity: muscle has only ever run `align` — enable `super5`/`progressive` for large or diverse families
- Use accumulated `strategy_stats.toon.gz` to bias RAG recommendations; retire high-failure strategies
- **Learned ESM threshold hygiene**: bands with lo < 0.10 ignored (prevents thr=0 spam); watch `esm_gap_stats.toon.gz` as KB grows
- **Indel preservation**: only gentle trims allowed — if gap still collapses to 0 without a trim, inspect ESM star-aligner gap handling

### Phase 2: Alignment Quality Predictor (Medium Priority)
**What:** Use pLM perplexity to predict alignment difficulty before running tools
**Why:** Smarter tool selection — hard sequences need MUSCLE/strategies, easy ones MAFFT auto
**How:**
- Compute per-residue perplexity from ESM-2
- High perplexity = hard to align = use MUSCLE or advanced MAFFT strategies
- Low perplexity = conserved = MAFFT auto sufficient
- Store perplexity in experiment records for future runs

### Phase 3: pLM-Based Alignment Scoring (Low Priority)
**What:** Use ProtT5 embeddings as alignment scoring (like PEbA)
**Why:** Better quality assessment than gap_fraction alone
**How:**
- Compute embedding similarity between aligned sequences
- Use as additional metric alongside sp_score, gap_fraction
- Could replace or augment extended_metrics

### Phase 4: Hybrid Tool Selection (Low Priority)
**What:** Pre-select MAFFT vs MUSCLE vs ESM-2 based on embedding distance
**Why:** Avoid wasting iterations on wrong tool
**How:**
- Compute pairwise embedding distances
- If average distance > threshold → likely divergent → ESM-2 or MUSCLE
- If average distance < threshold → likely conserved → MAFFT auto
- Train threshold on populated knowledge base

### Phase 5: Better Embeddings for Knowledge Base (Medium Priority)
**What:** Use ESM-2 sequence embeddings for warm start matching (alongside TF-IDF)
**Why:** Semantic embeddings find truly homologous proteins, not just k-mer similarity
**How:**
- Store ESM-2 mean-pooled embedding in experiment records
- Use for warm start matching (protein-only)
- Keep TF-IDF as fallback for nucleotide sequences
- Blended similarity: 0.7 * ESM-2 + 0.3 * TF-IDF

### Phase 6: Parallel Batch Processing (Medium Priority)
**What:** Process multiple families simultaneously using multiprocessing
**Why:** Speed up knowledge base building (currently ~2-3 min per family)
**How:**
- Use Python multiprocessing or concurrent.futures
- Limit to N concurrent processes based on CPU cores
- Each process gets its own process group for cleanup
- Progress tracking across all processes

### Phase 7: Web Dashboard (Low Priority)
**What:** Browser-based interface for monitoring batch builds
**Why:** Better visibility into long-running processes
**How:**
- Flask or Streamlit app
- Real-time progress updates via WebSocket
- Family statistics and charts
- Log viewing and filtering

## Current Pipeline Capabilities

| Feature | Status |
|---------|--------|
| Hill numbers (q0, q1, q2) | ✅ Implemented |
| ESM-2 embedding computation | ✅ Implemented |
| ESM-2 alignment (NW + star) | ✅ Implemented |
| ESM-2 as pipeline tool | ✅ Implemented |
| Hill numbers in LLM prompt | ✅ Implemented |
| Safe batch builder | ✅ Implemented |
| Timeout enforcement | ✅ Implemented |
| Resume support | ✅ Implemented |
| Min sequences filter | ✅ Implemented |
| Pfam data processing | ✅ Implemented |
| Gentle trimal only (indels kept) | ✅ Implemented |
| ESM gap-band threshold (lo≥0.10) | ✅ Implemented |
| build.sh fresh/resume launcher | ✅ Implemented |
| ESM-2 perplexity prediction | ❌ Not implemented |
| ProtT5 alignment scoring | ❌ Not implemented |
| Hybrid tool selection | ❌ Not implemented |
| ESM-2 embeddings for warm start | ❌ Not implemented |
| Parallel batch processing | ❌ Not implemented |
| Web dashboard | ❌ Not implemented |

## Dependencies Required

```
# Already installed in AI env
pip install numpy scipy torch transformers

# For parallel batch processing (Phase 6)
# No new dependencies needed (uses built-in multiprocessing)

# For web dashboard (Phase 7)
pip install flask  # or streamlit
```

## Notes

- ESM-2 650M is sufficient for alignment and embeddings
- ProtT5-XL is better for alignment tasks but 4x larger
- Pfam sequences (protein families) should have good conservation = low gap fraction
- Nucleotide sequences will continue using TF-IDF (no good pLM for nucleotides yet)
- Populating the knowledge base (all 30,007 families) is highest priority
- Safe batch builder enables long-running processes without zombie processes
