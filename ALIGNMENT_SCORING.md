# Alignment Scoring: Good vs Bad

How the MSA pipeline decides whether an alignment is **good**, **moderate**, or **bad** — using gap fraction, Shannon entropy / conservation, Hill numbers (q0/q1/q2), Sum-of-Pairs, identity, and related parameters.

All formulas and thresholds below match the code (primary sources: `scripts/extended_metrics.py`, `scripts/pipeline_loop.py`, `scripts/ai_controller.py`, `scripts/rag_store.py`, `scripts/esm_analysis.py`).

---

## 1. Overview

Every iteration computes **14 metrics** via `compute_extended_metrics()`:

| Group | Metrics |
|-------|---------|
| **Basic** | `gap_fraction`, `alignment_length`, `avg_sequence_length` |
| **Extended** | `sp_score`, `cs_score`, `conservation`, `gap_distribution`, `identity` |
| **Hill numbers** | `hill_q0_avg`, `hill_q1_avg`, `hill_q2_avg`, `hill_q0_max`, `hill_q1_max`, `hill_problematic_columns` |

**Primary quality gate is `gap_fraction`.** Hill numbers, Shannon conservation, SP/CS, and identity refine decisions (tool switch, trim, accept, ESM probe).

---

## 2. Primary gate: `gap_fraction`

```
gap_fraction = (number of '-' characters) / (total characters in alignment)
```

Lower is better. This single value drives accept/refine/trim/switch and problem-type labels.

| Threshold | Meaning in pipeline |
|-----------|---------------------|
| **`< 0.1`** | **Good** — eligible for accept (if also `sp > 0.5`) |
| **`< 0.15`** | After trim: auto-accept; **never accept** if `> 0.15` (hard LLM constraint) |
| **`0.1 – 0.15`** | Moderate / general — keep refining |
| **`> 0.15`** | **Bad alignment** — force tool switch or gentle trim; overrides accept/converged |
| **`> 0.2`** | High gap problem type (store/RAG) |
| **`0.12 – 0.40`** | ESM-2 **probe band** (try esm2 once) |
| **`> 0.25` + identity `< 0.2`** | Switch to ESM-2 |

**Accept rule (high confidence):**

```text
gap_fraction < 0.1  AND  sp_score > 0.5  →  accept
```

**Hard constraint (LLM + rules):**

```text
NEVER accept if gap_fraction > 0.15
```

---

## 3. Shannon entropy and conservation

### 3.1 Column Shannon entropy → `conservation`

Computed in `compute_conservation()` (`extended_metrics.py:181-214`):

1. For each alignment column, **drop gaps**.
2. Shannon entropy (base 2):

\[
H = -\sum_i p_i \log_2 p_i
\]

3. Max entropy for that column:

\[
H_{max} = \log_2(\min(\#\text{distinct residues},\ 20))
\]

4. Normalized conservation for the column: \(1 - H / H_{max}\)
5. Average over all valid columns → **`conservation`**.

| Value | Interpretation |
|-------|----------------|
| **→ 1.0** | Fully conserved column(s) (one residue dominates) |
| **→ 0.0** | Maximum diversity / no conservation |
| **Higher better** | Prompt: *“conservation: … (1-entropy, higher better)”* |

Gaps are **excluded** from entropy (only residues count).

### 3.2 ESM-2 per-residue entropy (separate path)

In `protein_embeddings.py`, ESM softmax probabilities give:

```
entropy = -Σ p log p          (natural log)
conservation_score = 1 - entropy / log(num_aa)
```

Stored per position; column-averaged as `esm_entropy` in `esm_analysis.py`. Used to compare **model-implied** conservation vs **observed** Hill diversity (section 9).

---

## 4. Hill numbers (q0, q1, q2)

Implemented in `extended_metrics.py:28-123`. Multi-scale **column diversity** (gaps excluded; columns with &lt; 2 residues → 0).

### Formulas

For residue counts \(c_i\) in a column, total \(N = \sum c_i\):

| Order | Name | Formula | Meaning |
|------:|------|---------|---------|
| **q = 0** | Richness | number of distinct residues | How many residues(Type) appear |
| **q = 1** | True diversity | \(\exp(H_{natural})\) = exp(Shannon, nats) | Effective number of equally common residues |
| **q = 2** | Inverse Simpson | \(\left(\sum p_i^2\right)^{-1}\) | Down-weights rare residues |

Code for q=1 (`extended_metrics.py:32-38`):

```python
entropy -= p * math.log(p)   # natural log
return math.exp(entropy)     # q=1 Hill number
```

### Interpretation (as used by the controller)

| Metric | Ideal / good | Diverse / problematic |
|--------|--------------|------------------------|
| `hill_q0_avg` | **1** = fully conserved | **> 5** = very diverse |
| `hill_q1_avg` | **1** = fully conserved | **> 10** → consider **esm2**; **> 5** = high diversity |
| `hill_q2_avg` | **1** = fully conserved | **> 5** = very diverse |

Averages (`hill_q*_avg`) are plain means over columns; `hill_q0_max` / `hill_q1_max` are column maxima.

### Problematic columns

A column is listed in `hill_problematic_columns` (capped at **10**) if:

```text
hill_q1 > 5.0   OR   column gap_fraction > 0.3
```

Each entry: `{column, hill_q1, gap_fraction, num_residues}`. First **5** are shown to the LLM in the prompt.

### Quick examples

| Column composition | q0 | q1 | q2 | Reading |
|--------------------|----|----|-----|---------|
| All `A` | 1 | 1 | 1 | Perfectly conserved |
| 5× `A`, 5× `L` | 2 | 2 | 2 | Balanced 50/50 |
| 8× `A`, 1× `L`, 1× `V` | 3 | ≈1.6 | ≈1.3 | Dominated by A |
| 10 distinct residues | 10 | 10 | 10 | Max diversity for n=10 |

---

## 5. Other parameters

| Metric | Formula / logic | Good direction | Typical “good” cue |
|--------|-----------------|----------------|--------------------|
| **`sp_score`** (Sum-of-Pairs) | Per column (gaps removed): fraction of matching residue pairs; average over valid columns | **Higher better** | **&gt; 0.5** with gap &lt; 0.1 → accept |
| **`cs_score`** (Column Score) | Fraction of columns where **all** residues are identical | **Higher better** | Ideal **&gt; 0.3** (prompt) |
| **`identity`** | Mean pairwise identity over non-gap positions, averaged across all pairs | Higher = more similar family | **&lt; 0.3** → diverse; **&lt; 0.2** + high gap → esm2 |
| **`gap_distribution`** | **Std dev** of per-column gap fractions | **Lower = gaps more evenly spread** | Even gaps preferred over all gaps in a few columns |
| **`instability`** (convergence) | `gap_diff + len_diff/len_scale` between iterations | **Lower better** | Accept docs: **&lt; 0.01** with gap &lt; 0.1; store labels **&gt; 0.1** unstable |
| **`alignment_length` / `avg_sequence_length`** | Length of MSA / mean seq length | Context only | — |

---

## 6. Decision rules: good vs bad actions

### 6.1 Rule engine (`pipeline_loop.py`, first match wins)

| # | Condition | Action |
|---|-----------|--------|
| 1 | `gap < 0.1` **and** `sp > 0.5` | **accept** (high confidence) |
| 2 | ESM probe: gap in **[0.12, 0.40]** (or above learned threshold), esm not tried | switch → **esm2** |
| 3 | `gap > 0.25` **and** `identity < 0.2` | switch → **esm2** |
| 4 | `gap > 0.15` **and** tool = mafft | switch → **muscle** |
| 5 | `gap > 0.15` **and** tool in (muscle, esm2) **and** streak ≥ 2 | **trim** `-gappyout` (gentle only) |
| 6 | `identity < 0.3` **and** `gap > 0.15` | switch → **muscle** |

### 6.2 Bad-alignment overrides (`pipeline_loop.py:315-341`)

```text
is_bad_alignment = gap_fraction > 0.15
```

- If bad: **accept → refine** (even if decision said accept)
- If bad but **converged**: **converged → trim**
- After trim: if `gap < 0.1` → auto-accept

### 6.3 LLM controller rules (`ai_controller.py:89-103`)

Same spirit, plus:

| Rule | Condition | Action |
|------|-----------|--------|
| 6 | `gap < 0.1` and `sp > 0.5` | **accept** |
| 7 | **`hill_q1_avg > 10`** | consider **esm2** |
| 8 | otherwise | **refine** (different strategy) |
| — | `gap > 0.15` | **NEVER accept** (hard constraint) |

Prompt ideals: **gap &lt; 0.1**, **sp &gt; 0.5**, **cs &gt; 0.3**, **conservation higher better**.

### 6.4 Outcome label after a run (`rag_store.py:24-30`)

| Outcome | Condition |
|---------|-----------|
| **`success`** | `final_gap < 0.1` |
| **`partial_improvement`** | `initial_gap > final_gap` (but not success) |
| **`no_improvement`** | otherwise |

---

## 7. Problem-type labels (three classifiers)

Labels feed RAG retrieval, external guidance, and dashboard grouping. **Thresholds differ slightly by path** — document all three.

### A. Live loop — `_classify_problem` (`pipeline_loop.py:122-135`)

| Label | Condition (order matters) |
|-------|---------------------------|
| `high_gap_fraction` | `gap > 0.2` **and** `sp < 0.3` |
| `diverse_sequences` | `identity < 0.3` |
| `moderate_gap_fraction` | `gap > 0.15` |
| `good_quality` | `gap < 0.1` **and** `sp > 0.5` |
| `general` | fallthrough |

### B. Stored on experiments — `_get_problem_type` (`rag_store.py:13-21`)

| Label | Condition |
|-------|-----------|
| `unstable` | `instability > 0.1` *(store path typically calls with instability=0, so rarely applied)* |
| `high_gap_fraction` | `gap > 0.2` |
| `moderate_gap_fraction` | `gap > 0.1` |
| `good_quality` | else (`gap ≤ 0.1`) |

**Note:** store uses **0.1** for moderate; live loop uses **0.15** for moderate and adds SP/identity checks.

### C. RAG fallback — `build_rag_context` (`rag_context.py:12-18`)

| Label | Condition |
|-------|-----------|
| `high_gap_fraction` | `gap > 0.2` |
| `moderate_gap_fraction` | `gap > 0.1` |
| `good_quality` | else |

### Dashboard order (`visualization.py` / `dashboard.py`)

```text
high_gap_fraction  →  moderate_gap_fraction  →  good_quality [reference]
```

`good_quality` is tagged **`[reference — any strategy works]`**.

---

## 8. Summary table: good vs bad

| Signal | **GOOD alignment** | **BAD / problematic** |
|--------|--------------------|------------------------|
| `gap_fraction` | **&lt; 0.1** (accept band) | **&gt; 0.15** (never accept); **&gt; 0.2** high_gap |
| `sp_score` | **&gt; 0.5** (with low gap) | **&lt; 0.3** with high gap → high_gap |
| `cs_score` | **&gt; 0.3** ideal | Near 0 → few fully conserved columns |
| `conservation` (1 − H/Hmax) | **Higher** (→ 1) | Low → diverse/noisy columns |
| `hill_q0/q1/q2_avg` | **≈ 1** (conserved) | **q1 &gt; 5** column-level; **q1_avg &gt; 10** → esm2 |
| `hill_problematic_columns` | Few / empty | Entries with **q1 &gt; 5** or **gap &gt; 0.3** |
| `identity` | Moderate–high (family-like) | **&lt; 0.3** diverse; **&lt; 0.2** + high gap → esm2 |
| `gap_distribution` | Low std (even gaps) | High std (gaps clustered in columns) |
| `instability` | **&lt; 0.01** (converged story) | **&gt; 0.1** unstable (store label) |
| Problem type | `good_quality` | `moderate_gap_fraction` / `high_gap_fraction` / `diverse_sequences` |
| Outcome | `success` (final gap &lt; 0.1) | `no_improvement` / `partial_improvement` |

---

## 9. ESM cross-checks (conservation vs Hill)

In `esm_analysis.analyze_conservation()` (`esm_analysis.py:81-133`):

| Concept | Definition |
|---------|------------|
| Observed diversity (mapped) | `1 - min(hill_q1 / 10, 1)` |
| **Correlation** | Pearson(ESM conservation, observed) |
| **Conserved column** | `esm_score > 0.7` |
| **Variable column** | `esm_score < 0.3` |
| **Misaligned column** | `esm_score > 0.7` **AND** `hill_q1 > 5.0` |

**Misaligned** = model says “highly conserved” but observed alignment shows high Hill diversity → likely wrong column placement.

Correlation bands (report): **&gt; 0.7** agrees, **&gt; 0.3** partial, else disagrees.

Optional gap refinement: position gap penalty  
`gap_open * (1 + alpha * conservation)` — conserved columns get higher open cost.

**ESM gap “success”** (post-correction / band learning):  
`(gap_before - gap_after) >= 0.02` **or** `gap_after < 0.15`.

---

## 10. Worked examples (from real run logs)

### Example A — Good accept (`logs/b96d7762.jsonl`)

| Metric | Value | Reading |
|--------|------:|---------|
| gap_fraction | **0.058** | &lt; 0.1 ✅ |
| sp_score | **0.781** | &gt; 0.5 ✅ |
| identity | 0.754 | similar family |
| hill_q1_avg | **1.54** | near-conserved ✅ |
| conservation | 0.194 | modest but positive |
| Decision | **accept** | `rule: gap=0.058<0.1, sp=0.781>0.5` |

### Example B — Bad / needs work (high gap + low SP)

Typical pattern for `high_gap_fraction`:

| Metric | Example | Reading |
|--------|--------:|---------|
| gap_fraction | ~0.18–0.40 | **&gt; 0.15** → never accept; probe esm2 if in band |
| sp_score | ~0.03–0.33 | weak column agreement |
| hill_q1_avg | often **&gt; 5** | diverse or misaligned columns |
| Likely actions | change_tool / trim `-gappyout` / esm2 | per rules 1–7 |

### Example C — Misaligned column flag

| ESM conservation | hill_q1 | Flag |
|-----------------:|--------:|------|
| 0.85 | 7.2 | **misaligned** (ESM conserved, observed diverse) |
| 0.85 | 2.0 | conserved, agrees |
| 0.20 | 7.0 | variable, agrees (both diverse) |

---

## 11. Code map

| Concern | File | Lines (approx.) |
|---------|------|-----------------|
| Hill q0/q1/q2, problematic cols | `scripts/extended_metrics.py` | 28–123 |
| SP / CS / conservation / identity / gap_distribution | `scripts/extended_metrics.py` | 126–265 |
| Aggregate 14 metrics | `scripts/extended_metrics.py` | 268–295 |
| Rule accept / switch / trim | `scripts/pipeline_loop.py` | 73–119 |
| Live problem classify | `scripts/pipeline_loop.py` | 122–135 |
| Bad-alignment override | `scripts/pipeline_loop.py` | 315–341 |
| LLM prompt ideals + rules 1–8 | `scripts/ai_controller.py` | 46–103 |
| Store problem type + outcome | `scripts/rag_store.py` | 13–30 |
| RAG problem-type fallback | `scripts/rag_context.py` | 12–18 |
| ESM vs Hill, misaligned cols | `scripts/esm_analysis.py` | 81–133 |
| ESM residue entropy/conservation | `scripts/protein_embeddings.py` | 79, 124–136 |
| Instability / convergence | `scripts/convergence.py` | 22–39 |
| Dashboard problem ordering | `scripts/visualization.py` | 13, 119 |

---

## 12. Where results are stored

| Store | Path | Relevant fields |
|-------|------|-----------------|
| Experiment KB | `knowledge/experiments.jsonl.gz` | `problem_type`, `initial_gap_fraction`, `final_gap_fraction`, `extended_metrics` (all 14 + Hill), `final_outcome`, `lesson` |
| Per-iteration logs | `logs/<experiment_id>.jsonl` | full `metrics`, `convergence`, `decision` each iter |
| Strategy stats | `knowledge/strategy_stats.toon.gz` | success/failure per action:strategy |
| ESM band stats | `knowledge/esm_gap_stats.toon.gz` | gap-band n/success |
| Dashboard export | `learning_report.txt` | gap reduction **by problem type** |

---

## Quick cheat-sheet

```text
GOOD:   gap < 0.1  AND  sp > 0.5          → accept
        hill_q1_avg ≈ 1, conservation higher better
        outcome success if final gap < 0.1

BAD:    gap > 0.15                         → never accept; refine or gentle trim
        gap > 0.2 (+ sp < 0.3)             → high_gap_fraction
        hill_q1 > 5 (column)               → problematic column
        hill_q1_avg > 10                   → try esm2
        ESM high + hill_q1 high            → likely misaligned column

TRIM:   only gentle strategies (-gappyout / -automated1) — never strip all indels
```
