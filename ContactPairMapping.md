# Contact-Pair Mapping — Ways to Improve Decisions

**Goal:** use residue–residue contact pairs so gaps/indels near structurally important regions are preserved, and decisions (trim / tool / accept) get a new signal.

---

## 1. Where a contact map can come from

| # | Source | Cost | Quality | Notes |
|---|--------|------|---------|-------|
| C1 | **ESM-2 attention maps** (`output_attentions=True` on existing `EsmModel`) | Medium (same forward pass, +VRAM for attention tensors) | Good (ESM-2 attention ≈ contacts, last-layer heads) | Reuses `protein_embeddings.py`; no new model |
| C2 | **MSA mutual information + APC** (co-evolution, computed from current alignment) | Low (numpy, CPU only) | OK for deep families (>~200 seqs), weak for shallow | Zero GPU, works even before ESM loads |
| C3 | **Residue-embedding cosine pairs** (off-diagonal similarity of per-residue ESM vectors) | Low (embeddings already computed in ESM paths) | Medium — more "similarity" than physical contact | Free when `--esm-analysis`/refine already ran |
| C4 | **DCA/PLM-DCA** (probabilistic) | High (iterative, slow) | Best | Not recommended now — batch is 30k families |
| C5 | **PDB/AlphaFold experimental** | External files | Best | Out of scope (no structures in repo) |

**Recommendation:** C2 always-on (cheap) + C1 when ESM already runs; C3 as fallback.

---

## 2. Uses (the actual "better decisions")

### U1 — Trim protection (preserve gaps near contact regions)

- Compute per-column **contact participation score**: `col_score(c) = Σ contact_prob(c, j)` over top partners (or count of pairs > threshold).
- **Before/after trimal:** never drop a column with `col_score` in top-K% unless `gap_fraction=1` (pure-gap column with no residue).
- Implementation: post-trim **column restore** — diff pre/post columns (trimal only deletes, never reorders) and re-insert protected columns. No trimal flag needed (gentle-only strategy order unchanged).

### U2 — Gap penalty in `esm_analysis.refine_gaps` (line ~237)

- Current: `gap_open × (1 + alpha × esm_conservation)`.
- Add contact term: `× (1 + beta × contact_participation)` → gaps opened inside contact cores cost more; gaps at loop/periphery (low contact) stay cheap.
- **Effect:** refine keeps indels where structure matters, sheds them where it doesn't — exactly "preserve gaps near contact regions".

### U3 — New metric: `contact_pair_score` (alignment quality)

- For each predicted pair (i, j) in native sequence numbering: check both residues are present (not gapped) in the alignment and their **alignment distance** ≤ D (e.g., ≤ 3 columns apart, no long gap between).
- `contact_pair_score = satisfied / total_pairs`.
- Add to `extended_metrics.py` (15th metric), logged in `logs/*.jsonl`.

### U4 — Decision-engine rules (new R7/R8, `pipeline_loop.py` L67–119)

- **R7:** `gap > 0.15 AND contact_pair_score ≥ 0.6` → trim is *risky* → prefer `change_tool` first; trim only if score stays ≥ 0.5 after dry-run.
- **R8:** `contact_pair_score < 0.4 AND tool=mafft` → aligner is breaking pairs → switch muscle/esm2.
- **Accept guard:** never `accept` if `contact_pair_score` dropped >0.15 from best-so-far (like the gap guard at L315–341).

### U5 — LLM context (`ai_controller` prompt + `rag_context`)

- Feed `contact_pair_score`, `protected_columns`, top broken pairs into the prompt → LLM sees structural cost of trim, not just gap/identity.
- Store score in experiment JSONL → RAG can recommend "tool X preserves contacts best on family class Y".

### U6 — Trim strategy selector

- High contact density (many top-K columns) → prefer `-automated1` (identity-aware) over `-gappyout` (gap-only view); swap order in `TRIM_STRATEGY_ORDER` per-family instead of global.

### U7 — Convergence signal (`convergence.py`)

- Add `Δ contact_pair_score` to instability (currently `dGap + dLen/len`): an alignment that keeps pairs stable across iterations is converging even if gap wiggles.

### U8 — ESM post misalignment refinement (`analyze_conservation`)

- Current: column flagged if `esm > 0.7 AND hill_q1 > 5`.
- Add: column flagged if its **top contact partners are misaligned/gapped** while the column itself is conserved → "orphan conserved column" → stronger trim/realign hint.

### U9 — Domain / block awareness

- Contact maps show **blocks** (domains). Gaps *between* blocks (linkers) are safe to trim; gaps *inside* blocks are not.
- Practical proxy: low intra-block contact columns = linker → allow `gap > 0.15` there without triggering R7.

### U10 — Warm-start / knowledge base

- Store per-family `contact_pair_score` + which tool/strategy preserved it → next similar family seeds strategy by **contact preservation**, not only final gap.

---

## 3. Suggested phasing (minimal first)

| Phase | Items | Risk |
|-------|-------|------|
| 1 | C2 MI map + U3 metric + log only (observe on current batch) | none — no decision change |
| 2 | U1 trim protection + U2 gap penalty term | gentle, preserves indels (aligned with gentle-only rules) |
| 3 | U4 rules + U5 LLM context (gated by flag `--contact-rules`) | changes decisions — test first |
| 4 | C1 ESM attention (better map), U6–U10 | later |

---

## 4. Code touchpoints (for later implementation)

- `scripts/extended_metrics.py` — U3 metric (next to Hill, L45+)
- `scripts/contact_map.py` *(new)* — C2/C3 map builder + column scores
- `scripts/trimal_runner.py` or `pipeline_loop.py` L343–350 — U1 restore step
- `scripts/esm_analysis.py:237` — U2 penalty formula
- `scripts/pipeline_loop.py:67–119` — U4 rules; `315–341` accept guard
- `scripts/ai_controller.py` prompt, `rag_store.py` — U5/U10
- CLI flags: `--contact-map mi|esm|off` (default `mi`), `--contact-protect on/off`

---

## 5. Honest limits

- MI needs enough sequences (shallow Pfam families → noisy map → skip U4 on `n < 100`).
- Attention maps add VRAM (T600 4GB already tight with Ollama) → ESM attention only when GPU free.
- Contact map is per-*sequence* pair (native numbering); after alignment it maps to columns only for residues present — gapped residues can't be scored (that's fine: gapped = unsatisfied pair).

---

Related: [FlowChart.md](FlowChart.md) (system map, Diagram B/C/D), [ALIGNMENT_SCORING.md](ALIGNMENT_SCORING.md) (metric thresholds).
