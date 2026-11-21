# MSA AI — Full Flowcharts

Complete map of how this AI works, split into **readable subsets**. Each diagram is self-contained; arrows between sections are shown in **Diagram 0**.

**Legend**

| Shape | Meaning |
|-------|---------|
| `([...])` | Entry / exit |
| `[[...]]` | Process step |
| `{...}` | Decision (threshold on edge) |
| `-->` | Main flow |
| `-.-` | Knowledge / context feed |
| `==>` | Outer cycle (next run / next family) |
| Subgraph | Module boundary |

| Color class | Meaning |
|-------------|---------|
| 🟦 tools | Alignment / trim execution |
| 🟨 decision | Rules, overrides, classify |
| 🟩 knowledge | RAG, stats, experiment store |
| 🟥 ESM | ESM-2 probe / align / post / analysis |
| 🟪 batch | Batch build / workers |

Primary files: `run_pipeline.py`, `scripts/pipeline_loop.py`, `scripts/ai_controller.py`, `scripts/extended_metrics.py`, `scripts/rag_retrieve.py`, `scripts/rag_store.py`, `scripts/build_knowledge.py`.

---

## Diagram 0 — Big picture

```mermaid
flowchart LR
  subgraph IN[Inputs]
    CLI([run_pipeline.py CLI])
    BATCH([build.sh / restartBuild.sh<br/>30k Pfam families])
  end

  subgraph CORE[Iteration core - see A]
    LOOP[[align → metrics → converge → decide → act]]
  end

  subgraph DEC[Decision engine - see B]
    RULES[Rule engine R1-R6]
    LLM[Ollama llama3]
    OVR[Post overrides]
  end

  subgraph TOOLS[Tools - see C]
    MAFFT[mafft]
    MUSCLE[muscle]
    ESMA[esm2 align]
    TRIM[trimal gentle]
  end

  subgraph ESM[ESM subsystem - see D]
    ESMPOST[probe / post / conservation]
  end

  subgraph KNOW[Knowledge - see E]
    READ[warm start + RAG + hints]
    WRITE[experiments.jsonl.gz + stats]
  end

  subgraph OUT[Reporting - see G]
    DASH[dashboard / learning_report]
    LOG[logs/ID.jsonl]
  end

  CLI --> LOOP
  BATCH -->|spawn workers| LOOP
  LOOP --> READ
  READ -.context.-> DEC
  RULES -->|miss| LLM
  DEC --> TOOLS
  TOOLS --> LOOP
  LOOP --> ESMPOST
  ESMPOST --> WRITE
  LOOP --> LOG
  WRITE ==>|next experiment| READ
  WRITE --> DASH
  BATCH -->|OK/FAILED| LOG
```

---

## A. Entry & iteration loop

```mermaid
flowchart TD
  START([run_pipeline.py]) --> ARGS[args: -n max_iter, -t tol, -p patience,<br/>-w warm 0.8, --esm-post if_bad]
  ARGS --> READF[read FASTA text]
  READF --> SETUP[[experiment_id = uuid8,<br/>ConvergenceTracker, state init<br/>tool=mafft --auto]]
  SETUP --> WS{warm start hit?<br/>sim ≥ 0.8}
  WS -->|Yes| SEED[seed tool/strategy<br/>avoid trims from plan]
  WS -->|No| HINT
  SEED --> HINT[esm post hint]
  HINT --> HEAD[iteration i = 1..max]

  HEAD --> ALIGN[[run_alignment<br/>tool → mafft/muscle/esm2]]
  ALIGN --> MET[[compute_extended_metrics<br/>14 metrics]]
  MET --> CONV[[ConvergenceTracker.check<br/>instability = dGap + dLen/len]]
  CONV --> CTX[assemble context:<br/>warm + RAG + external + hints]
  CTX --> DECIDE[DECISION - see diagram B]

  DECIDE --> ACT{action}
  ACT -->|accept / stop / converged| BREAK[break loop]
  ACT -->|trim success gap<0.1| BREAK
  ACT -->|refine / change_tool /<br/>change_strategy / trim pending| NEXT[i++]
  NEXT -->|i ≤ max| HEAD
  NEXT -->|i > max| BREAK

  BREAK --> SAVE[logger.save → logs/ID.jsonl]
  SAVE --> POST[ESM post gate - see D]
  POST --> STORE[[store_experiment - see E]]
  STORE --> RET([return result dict])
  RET --> PRINT[print FINAL RESULT<br/>iters, metrics, exp_id, preview]
```

**Exit conditions:** `accept` / `stop` / `converged` / trim-to-accept / `max_iterations` exhausted.

---

## B. Decision engine (full depth)

Per-iteration path: metrics → classify → **rules** → **LLM** → **fallback** → **overrides** → action.

```mermaid
flowchart TD
  IN[metrics + convergence + context] --> CLS{problem classify<br/>_classify_problem}

  CLS -->|gap>0.2 AND sp<0.3| P1[high_gap_fraction]
  CLS -->|identity<0.3| P2[diverse_sequences]
  CLS -->|gap>0.15| P3[moderate_gap_fraction]
  CLS -->|gap<0.1 AND sp>0.5| P4[good_quality]
  CLS -->|else| P5[general]

  P1 & P2 & P3 & P4 & P5 --> R1

  subgraph RULES[Rule engine - first match wins - pipeline_loop L67-119]
    R1{R1: gap<0.1 AND sp>0.5?}
    R1 -->|Yes| ACC[accept - confidence high]
    R1 -->|No| R2{R2: not esm_tried<br/>AND tool != esm2<br/>AND 0.12 ≤ gap ≤ 0.40?}
    R2 -->|Yes| ESM1[change_tool → esm2 local<br/>one-shot probe]
    R2 -->|No| R2B{R2b: learned thr<br/>AND gap > thr<br/>default 0.25?}
    R2B -->|Yes| ESM1
    R2B -->|No| R3{R3: gap>0.25<br/>AND identity<0.2<br/>AND tool != esm2?}
    R3 -->|Yes| ESM2[change_tool → esm2 local]
    R3 -->|No| R4{R4: gap>0.15<br/>AND tool=mafft?}
    R4 -->|Yes| MU[change_tool → muscle align]
    R4 -->|No| R5{R5: gap>0.15<br/>AND tool ∈ muscle,esm2<br/>AND streak ≥ 2?}
    R5 -->|Yes| TR[trim -gappyout]
    R5 -->|No| R6{R6: identity<0.3<br/>AND gap>0.15<br/>AND tool ∉ muscle,esm2?}
    R6 -->|Yes| MU
    R6 -->|No - no rule| MISS[rule returned None]
  end

  ACC --> OVR
  ESM1 --> OVR
  ESM2 --> OVR
  MU --> OVR
  TR --> OVR

  MISS --> PROMPT

  subgraph LLM[LLM path - ai_controller.decide_action]
    PROMPT[[build prompt:<br/>metrics, convergence, RAG text,<br/>tool catalogs, rules, examples, JSON schema]]
    PROMPT --> CALL[POST Ollama /api/generate<br/>llama3, num_predict=200, 120s]
    CALL --> VAL{validate ≤3 tries:<br/>action/tool/mode whitelist<br/>sanitize extra_args<br/>metrics echo ±0.05}
    VAL -->|invalid| RETRY{retry left?}
    RETRY -->|Yes| CALL
    RETRY -->|No| FAILLLM[ValueError]
    VAL -->|OK| LLMDEC[action JSON]
  end

  FAILLLM --> FB
  CALL -->|timeout/error| FB

  subgraph FALL[Fallback - on LLM failure]
    FB{gap < 0.15?}
    FB -->|Yes| FBACC[accept - low confidence]
    FB -->|No| FBTR[trim -gappyout - low confidence]
  end

  LLMDEC --> OVR
  FBACC --> OVR
  FBTR --> OVR

  subgraph OVER[Post-decision overrides - pipeline_loop L304-341]
    OVR[[update tool_streak<br/>no_improvement if gap not -0.005]]
    OVR --> SW{auto-switch:<br/>gap>0.15 AND streak≥2<br/>AND action ∈ refine,change_strategy,accept?}
    SW -->|Yes| FORCE[force change_tool<br/>mafft → muscle → esm2 → cycle]
    SW -->|No| OV1{action=accept<br/>AND gap>0.15?}
    OV1 -->|Yes| REF[force refine<br/>NEVER accept bad gap]
    OV1 -->|No| OV2{converged<br/>AND gap>0.15?}
    OV2 -->|Yes| TRIMC[force trim<br/>converged=False]
    OV2 -->|No| OUT[action final]
  end

  FORCE --> OUT
  REF --> OUT
  TRIMC --> OUT
  OUT --> DONE([dispatch → diagram C<br/>then log_step → loop A])
```

### B — LLM rules & hard constraints (prompt, `ai_controller.py:89-103`)

| # | Condition | Action |
|---|-----------|--------|
| 1 | `gap > 0.15` and tool=mafft | change_tool → muscle |
| 2 | `gap > 0.15` and tool=muscle | trim `-gappyout` (gentle only) |
| 3 | `gap > 0.15` and tool=esm2 | trim `-gappyout` |
| 4 | `gap > 0.25` and `identity < 0.2` | change_tool → esm2 |
| 5 | `identity < 0.3` and `gap > 0.15` | change_tool → muscle |
| 6 | `gap < 0.1` and `sp > 0.5` | **accept** |
| 7 | `hill_q1_avg > 10` | consider esm2 |
| 8 | otherwise | refine |

**Hard constraints:** never accept if `gap > 0.15`; switch tool after 2 failures; trimal args only `-gappyout`/`-automated1` (never `-strict`/`-strictplus`/`-nogaps`).

### B — Validation pipeline (3 attempts)

```text
response JSON
  → _validate_action   (whitelist action/tool/msa_mode; bad → coerce refine/mafft)
  → _sanitize_extra_args (per-tool flag allowlist; trim → gappyout/automated1 only)
  → _check_metrics_echo  (echoed gap/sp/identity within 0.05 of actual)
  → fail: retry ≤ 3 → ValueError → fallback (FB)
```

### B — Fallback thresholds

| Condition | Result |
|-----------|--------|
| LLM exception / timeout / 3 invalid | `gap < 0.15` → **accept** (low conf) |
| else | **trim `-gappyout`** (low conf) |

---

## C. Tool execution & action dispatch

```mermaid
flowchart TD
  ACT[action from B] --> D{tool?}
  D -->|mafft| MSA[[msa_runner.run_msa<br/>seqTools env<br/>strategies: auto/localpair/<br/>globalpair/genafpair/retree]]
  D -->|muscle| MU2[[muscle_runner.run_muscle<br/>align/progressive/super5<br/>maxiters ≥ 1000]]
  D -->|esm2| ES[[esm_aligner.run_esm_alignment<br/>see diagram D]]

  MSA & MU2 & ES --> DONE2[new alignment]

  ACT --> A{action type}
  A -->|accept/stop/converged| EXIT([break loop])
  A -->|trim| PICK[_pick_trim_strategy<br/>-gappyout then -automated1<br/>gentle only]
  PICK --> RUNTR[[trimal_runner.run_trimal]]
  RUNTR --> REMET[recompute metrics]
  REMET --> RST[reset ConvergenceTracker]
  RST --> OKGAP{gap < 0.1?}
  OKGAP -->|Yes| ACC2[accept + break]
  OKGAP -->|No| REF2[action=refine, clear extra_args]
  REF2 --> LOOPBACK
  A -->|change_tool| SETT[set tool + strategy map<br/>esm2 → local, muscle → align<br/>esm_tried=True]
  A -->|change_strategy| SETS[strategy = --mode]
  A -->|refine| CYCLE[mafft cycle:<br/>auto → localpair → globalpair →<br/>genafpair → retree 1]
  SETT & SETS & CYCLE --> LOOPBACK([back to loop A])
  DONE2 --> LOOPBACK
  EXIT --> POSTX([ESM post - D])
```

---

## D. ESM subsystem (4 paths)

```mermaid
flowchart TD
  subgraph P1[1 - In-loop probe]
    GAP{gap in 0.12-0.40<br/>or gap > learned thr?}
    GAP -->|Yes, not tried| PROBE[change_tool → esm2 one-shot<br/>esm_tried=True]
  end

  subgraph P2[2 - ESM alignment]
    FA[FASTA] --> EMB[[compute_residue_embeddings<br/>EsmModel local cache]]
    EMB --> STAR[[star align:<br/>center = max cosine sum,<br/>NW gap_open=-2 extend=-0.5]]
    STAR --> EOUT[aligned FASTA]
  end

  subgraph P3[3 - ESM post gate - after loop]
    G{esm_post mode}
    G -->|always| RUN
    G -->|if_bad - default| GATE{gap > 0.12<br/>AND advice != avoid?}
    G -->|off| SKIP[skip]
    GATE -->|Yes| RUN
    GATE -->|No| SKIP
    RUN[[run_esm_analysis]]
    RUN --> PROB[[MLM probs → per-residue<br/>entropy / conservation]]
    PROB --> ANA[analyze_conservation:<br/>Pearson r ESM vs 1-minHill/10]
    ANA --> MIS{esm>0.7<br/>AND hill_q1>5?}
    MIS -->|Yes| FLAG[misaligned column<br/>report: re-align or trim]
    MIS -->|No| REPC[column report]
    RUN -.->|--esm-refine-gaps| REFINE[[refine_gaps alpha=2:<br/>gap_open × 1+alpha×cons<br/>NW re-run]]
    REFINE --> NEWGAP[new alignment]
    NEWGAP --> SUCC{dGap ≥ 0.02<br/>OR after < 0.15?}
    SUCC -->|Yes| EOK[post success]
    SUCC -->|No| EFAIL[post weak]
  end

  PROBE --> P2
  SKIP --> ST([store - E])
  EOK --> ST
  EFAIL --> ST
  REPC --> ST
```

**Notes**

- Learned ESM threshold: lowest 0.05-gap band with `n≥3` and success ≥50%, bands `lo≥0.10` only; default `(0.25, not learned)`.
- **No co-residue map:** decisions are per-column/per-position only; neighbors never cross-checked explicitly.
- Model: `models/models--facebook--esm2_t33_650M_UR50D` (local HF snapshot).

---

## E. Knowledge & RAG loop

```mermaid
flowchart LR
  subgraph READ[Read - per iteration, pre-decision]
    WARM[[find_warm_start<br/>TF-IDF sim ≥ 0.8 → derive_warm_plan]]
    RAGC[[build_rag_context top_k=3<br/>problem from gap: >0.2 high, >0.1 mod, else good]]
    WREC[[get_weighted_recommendations<br/>confidence × time decay, top-5]]
    EXT[[get_external_context<br/>MAFFT/TRIMAL docs]]
    HINT2[esm gap + post hints]
    WARM & RAGC & WREC & EXT & HINT2 --> CTXB[context → diagram B]
  end

  subgraph WRITE[Write - post-loop]
    ST0[store_experiment]
    ST0 --> SIG[k=4 signature + TF-IDF emb + fingerprint]
    SIG --> DUP{dup sim ≥ 0.7?}
    DUP -->|Yes| SKIPD[skip store]
    DUP -->|No| OUTC[outcome:<br/>final<0.1 success,<br/>init>final partial,<br/>else no_improvement]
    OUTC --> GZ[[append experiments.jsonl.gz]]
    GZ --> IDX[update tfidf_index.npz]
    IDX --> SST[strategy_stats.toon.gz]
    SST --> EG[esm_gap_stats 0.05 bands]
    EG --> EP[esm_post_stats]
    EP --> PROC[processed_families.txt.gz]
  end

  CTXB --> RUNN[run experiment N]
  RUNN --> ST0
  PROC ==>|next family| READ
  GZ ==>|next experiment N+1| WARM
  EG -.band advice.-> HINT2
```

**Dedup:** skip if near-duplicate (TF-IDF ≥ 0.7). **ESM band success:** `Δgap ≥ 0.02 OR after < 0.15`.

---

## F. Batch build

```mermaid
flowchart TD
  subgraph LAUNCH[Launcher]
    BS[build.sh<br/>fresh: wipe knowledge+logs<br/>--resume: keep]
    RB[restartBuild.sh 6 phases:<br/>1 kill batch  2 restart ollama<br/>3 VRAM ≥ 1500  4 torch CUDA check<br/>5 smoke --limit 2 ≥1 OK  6 launch]
    RB --> LAUNCH2[full build]
    BS --> LAUNCH2
  end

  LAUNCH2 --> MAIN[[build_knowledge.main<br/>scan 30007 *.faa, sort by seqs]]
  MAIN --> RES{--resume?}
  RES -->|Yes| SKIP2[skip processed_families.txt.gz]
  RES -->|No| LOOPF
  SKIP2 --> LOOPF[for each family]

  LOOPF --> MIN{seqs ≥ min 10?}
  MIN -->|No| SKIPI[SKIP too few]
  MIN -->|Yes| SUB[max-seqs 8000 subsample]
  SUB --> SPAWN[multiprocessing spawn child<br/>os.setpgrp, run_pipeline<br/>tol=0.001, patience=3, esm_post=if_bad]

  SPAWN --> JT{join timeout 300s}
  JT -->|alive| KILL[killpg SIGKILL → TimeoutError]
  JT -->|done| RES2{__success?}
  KILL --> FAILT[FAILED: timeout]
  RES2 -->|No, __error| FAIL2[FAILED: error / signal]
  RES2 -->|Yes| OK[OK: file → exp_id<br/>mark_family_processed]
  OK & FAILT & FAIL2 --> NEXTF[next family]
  NEXTF --> LOOPF
  LOOPF -->|end / SIGINT| SUM[summary → batch_knowledge.log]

  SPAWN -.child.-> PIPE[run_pipeline - diagrams A-E]
  PIPE -.results.-> RES2
```

**Ops notes:** PID → `logs/batch_knowledge.pid`; log → `logs/batch_knowledge.log`. Known hang pattern: child stuck in CUDA → log silent; fix = kill + `restartBuild.sh --resume` (sidecar keeps progress).

---

## G. Reporting & outputs

```mermaid
flowchart LR
  STEPS[log_step per iter] --> JL[logs/experiment_id.jsonl]
  STEPS --> PROC2[knowledge files - E]
  PROC2 --> DASH[[dashboard.py flags]]
  DASH --> D1[--dashboard: gap by problem type,<br/>tool usage, strategy table,<br/>learning curve, esm stats]
  DASH --> D2[--report → learning_report.txt]
  DASH --> D3[--export / --validate<br/>k-fold accuracy + lift]
  DASH --> D4[--feedback → user_feedback.json]
  BATCHLOG[batch_knowledge.log] --> D1
  D1 --> VIZ[stdout tables]
  D2 --> FILE[learning_report.txt]
```

**Key outputs**

| Output | Path |
|--------|------|
| Step logs | `logs/<experiment_id>.jsonl` |
| Experiments | `knowledge/experiments.jsonl.gz` |
| Stats | `knowledge/strategy_stats.toon.gz`, `esm_gap_stats.toon.gz`, `esm_post_stats.toon.gz` |
| Index | `knowledge/tfidf_index.npz`, `tfidf_index_vocab.json` |
| Batch | `logs/batch_knowledge.log`, `.pid`, `smoke_test.log` |
| Report | `learning_report.txt` |

---

## Cross-cutting cycles

| # | Cycle | Where |
|---|-------|-------|
| 1 | Iteration: metrics → decide → act → tool → log → metrics | A ⇄ B ⇄ C |
| 2 | Trim sub-cycle: trim → recompute → reset → accept or refine | C |
| 3 | Tool auto-switch: mafft → muscle → esm2 → mafft | B overrides |
| 4 | Learning across runs: store → next warm start / hints / LLM context | E ⇄ B |
| 5 | Batch: family → child pipeline → store → next family | F |
| 6 | ESM post overwrites last metrics before store (KB sees post gap) | D → E |

---

## File ↔ diagram index

| Diagram | Main files |
|---------|------------|
| 0 Big picture | all |
| A Entry & loop | `run_pipeline.py`, `scripts/pipeline_loop.py` |
| B Decision engine | `pipeline_loop.py` (rules/overrides), `ai_controller.py`, `llm_core.py` |
| C Tools | `msa_runner.py`, `muscle_runner.py`, `trimal_runner.py`, `esm_aligner.py`, `convergence.py` |
| D ESM | `esm_aligner.py`, `esm_analysis.py`, `protein_embeddings.py` |
| E Knowledge | `rag_store.py`, `rag_retrieve.py`, `rag_context.py`, `rag_advanced.py`, `knowledge_base.py`, `tfidf_sparse.py` |
| F Batch | `build.sh`, `restartBuild.sh`, `scripts/build_knowledge.py` |
| G Reporting | `dashboard.py`, `visualization.py`, `logger.py` |

Related: [ALIGNMENT_SCORING.md](ALIGNMENT_SCORING.md) — good/bad thresholds (Hill, Shannon, gap).
