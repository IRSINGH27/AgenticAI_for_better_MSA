import json
from typing import Any, Optional
from llm_core import ask_llm_json
from mafft_config import MAFFT_STRATEGIES, MAFFT_COMMON_FLAGS
from trimal_runner import TRIMAL_STRATEGIES, TRIMAL_COMMON_FLAGS
from muscle_runner import MUSCLE_ALGORITHMS, MUSCLE_COMMON_FLAGS, MUSCLE_PERM_OPTIONS
from esm_aligner import ESM_STRATEGIES, ESM_FLAGS


def _build_prompt(metrics: dict, convergence: dict, extra_flags: dict) -> str:
    mafft_strategies = "\n".join([f"  {k}: {v}" for k, v in MAFFT_STRATEGIES.items()])
    mafft_flags = "\n".join([f"  {k}: {v}" for k, v in MAFFT_COMMON_FLAGS.items()])
    trimal_strategies = "\n".join([f"  {k}: {v}" for k, v in TRIMAL_STRATEGIES.items()])
    trimal_flags = "\n".join([f"  {k}: {v}" for k, v in TRIMAL_COMMON_FLAGS.items()])
    muscle_algorithms = "\n".join([f"  {k}: {v}" for k, v in MUSCLE_ALGORITHMS.items()])
    muscle_flags = "\n".join([f"  {k}: {v}" for k, v in MUSCLE_COMMON_FLAGS.items()])
    muscle_perm = "\n".join([f"  {k}: {v}" for k, v in MUSCLE_PERM_OPTIONS.items()])
    esm_strategies = "\n".join([f"  {k}: {v}" for k, v in ESM_STRATEGIES.items()])
    esm_flags = "\n".join([f"  {k}: {v}" for k, v in ESM_FLAGS.items()])

    hill_str = ""
    hill_q0 = metrics.get("hill_q0_avg", 0)
    hill_q1 = metrics.get("hill_q1_avg", 0)
    hill_q2 = metrics.get("hill_q2_avg", 0)
    hill_q1_max = metrics.get("hill_q1_max", 0)
    hill_problematic = metrics.get("hill_problematic_columns", [])
    if hill_q0 > 0:
        hill_str = f"""- hill_q0_avg: {hill_q0:.3f}  (avg distinct residues per column, 1=conserved, >5=very diverse)
- hill_q1_avg: {hill_q1:.3f}  (avg true diversity, 1=conserved, >10=very diverse)
- hill_q2_avg: {hill_q2:.3f}  (avg inverse Simpson, 1=conserved, >5=very diverse)
- hill_q1_max: {hill_q1_max:.3f}  (most diverse column)"""
        if hill_problematic:
            cols = ", ".join([f"col{c['column']}(d={c['hill_q1']:.1f},gap={c['gap_fraction']:.0%})" for c in hill_problematic[:5]])
            hill_str += f"\n- problematic columns: {cols}"

    extra_args_str = ""
    if extra_flags:
        extra_args_str = f"\nCurrent MAFFT extra args: {extra_flags.get('extra_args', '')}"
        if extra_flags.get('trimmed'):
            extra_args_str += f"\nAlignment was already trimmed with trimal"
        if extra_flags.get('current_tool'):
            extra_args_str += f"\nCurrent tool: {extra_flags.get('current_tool')}"

    rag_context = extra_flags.get("rag_context", "")

    prompt = f"""You are an MSA (Multiple Sequence Alignment) controller with multiple alignment tools.

CURRENT ALIGNMENT METRICS:
- gap_fraction: {metrics.get('gap_fraction', 0):.4f}  (lower is better, ideal < 0.1)
- alignment_length: {metrics.get('alignment_length', 0)}
- avg_sequence_length: {metrics.get('avg_sequence_length', 0):.2f}
- sp_score: {metrics.get('sp_score', 0):.4f}  (Sum-of-Pairs, higher better, ideal > 0.5)
- cs_score: {metrics.get('cs_score', 0):.4f}  (Column Score, fraction conserved, ideal > 0.3)
- conservation: {metrics.get('conservation', 0):.4f}  (1-entropy, higher better)
- identity: {metrics.get('identity', 0):.4f}  (avg pairwise identity, higher better)
- gap_distribution: {metrics.get('gap_distribution', 0):.4f}  (gap clustering, lower=more even)
{hill_str}

CONVERGENCE STATUS:
- status: {convergence.get('status', 'unknown')}
- instability: {convergence.get('instability', 1.0):.4f}{extra_args_str}

{rag_context}

AVAILABLE TOOLS:

1. MAFFT (alignment - most popular, accurate):
{mafft_strategies}
MAFFT FLAGS:
{mafft_flags}

2. MUSCLE (alignment - fast, good for diverse sequences):
{muscle_algorithms}
MUSCLE PERMUTATION OPTIONS (-perm):
{muscle_perm}
MUSCLE FLAGS:
{muscle_flags}

3. ESM-2 (alignment - deep learning embeddings, best for divergent proteins):
{esm_strategies}
ESM-2 FLAGS:
{esm_flags}

4. TRIMAL (trimming - removes poorly aligned columns):
{trimal_strategies}
TRIMAL FLAGS:
{trimal_flags}

DECISION RULES (apply in order, first match wins):
1. If gap_fraction > 0.15 AND current tool is mafft: action="change_tool", tool="muscle", msa_mode="align"
2. If gap_fraction > 0.15 AND current tool is muscle: action="trim" with extra_args="-gappyout" (gentle — never use -strict/-strictplus/-nogaps; they delete indel columns)
3. If gap_fraction > 0.15 AND current tool is esm2: action="trim" with extra_args="-gappyout" (same gentle rule)
4. If gap_fraction > 0.25 AND identity < 0.2: action="change_tool", tool="esm2", msa_mode="local"
5. If identity < 0.3 AND gap_fraction > 0.15: action="change_tool", tool="muscle", msa_mode="align"
6. If gap_fraction < 0.1 AND sp_score > 0.5: action="accept"
7. If hill_q1_avg > 10: consider tool="esm2"
8. Otherwise: action="refine" with a different strategy

HARD CONSTRAINTS:
- NEVER accept if gap_fraction > 0.15
- If current tool has been used 2+ times without improvement, you MUST switch tools
- Do NOT stay on mafft if gap_fraction > 0.15 — switch to muscle immediately
- extra_args must be valid for the chosen tool: NEVER pass MAFFT flags (--maxiterate, --op, --ep, --nuc, --amino) to muscle; NEVER pass muscle flags to mafft; for trim action extra_args must be "-gappyout" or "-automated1" only (NEVER -strict, -strictplus, or -nogaps — they remove indel/gap columns)

TOOL SWITCHING GUIDANCE:
- MAFFT: default, good for conserved sequences (identity > 0.5)
- MUSCLE: better for diverse sequences (identity < 0.4, high gap fraction)
- ESM-2: best for highly divergent proteins (identity < 0.2) using deep learning embeddings

EXAMPLES:

Example 1 — High gap on MAFFT:
Metrics: gap_fraction=0.22, sp_score=0.25, identity=0.35, current_tool=mafft
Correct response:
{{"action": "change_tool", "tool": "muscle", "msa_mode": "align", "extra_args": "", "reason": "gap_fraction 0.22 > 0.15, switching from mafft to muscle", "confidence": "high", "metrics_echo": {{"gap_fraction": 0.22, "sp_score": 0.25, "identity": 0.35}}}}

Example 2 — Good alignment:
Metrics: gap_fraction=0.05, sp_score=0.72, identity=0.65, current_tool=mafft
Correct response:
{{"action": "accept", "tool": "mafft", "msa_mode": "auto", "extra_args": "", "reason": "gap_fraction 0.05 < 0.1, sp_score 0.72 > 0.5, alignment is good", "confidence": "high", "metrics_echo": {{"gap_fraction": 0.05, "sp_score": 0.72, "identity": 0.65}}}}

Example 3 — Stable but poor, try trimming:
Metrics: gap_fraction=0.18, sp_score=0.45, identity=0.40, current_tool=muscle, instability=0.0
Correct response:
{{"action": "trim", "tool": "muscle", "msa_mode": "align", "extra_args": "-gappyout", "reason": "stable but gap_fraction 0.18 > 0.15, mild trim with trimal -gappyout", "confidence": "high", "metrics_echo": {{"gap_fraction": 0.18, "sp_score": 0.45, "identity": 0.40}}}}

OUTPUT JSON with keys:
- action: accept | refine | trim | change_tool | change_strategy | stop
- tool: mafft | muscle | esm2 (which tool to use)
- msa_mode: auto | localpair | globalpair | fast (for MAFFT) | align | progressive | super5 (for MUSCLE) | local | global (for ESM-2)
- extra_args: string of flags (depends on action)
- reason: short explanation
- confidence: high | medium | low
- metrics_echo: {{"gap_fraction": <value from prompt>, "sp_score": <value from prompt>, "identity": <value from prompt>}}
"""
    return prompt


def _build_retry_prompt(original_prompt: str, last_response: Any) -> str:
    return original_prompt + "\n\nPrevious response was invalid. Respond with ONLY valid JSON."


def _sanitize_extra_args(action: str, tool: str, extra_args: str) -> str:
    if not extra_args:
        return ""

    tokens = extra_args.split()

    if action == "trim":
        valid_first = set(TRIMAL_STRATEGIES.keys())
        valid_all = valid_first | set(TRIMAL_COMMON_FLAGS.keys())
        kept = []
        i = 0
        while i < len(tokens):
            tok = tokens[i]
            if tok in valid_all:
                kept.append(tok)
                if tok not in valid_first and i + 1 < len(tokens) and not tokens[i + 1].startswith("-"):
                    kept.append(tokens[i + 1])
                    i += 2
                    continue
            i += 1
        if not kept or kept[0] not in valid_first:
            return ""
        return " ".join(kept)

    if tool == "muscle":
        allowed = {
            "-perm", "-perturb", "-replicates", "-threads", "-quiet",
            "-diversified", "-stratified", "-maxcc",
            "-minconf", "-maxgapfract",
        }
    elif tool == "mafft":
        allowed = set(MAFFT_COMMON_FLAGS.keys()) | set(MAFFT_STRATEGIES.keys())
        allowed |= {"--maxiterate", "--thread", "--quiet", "--reorder"}
    else:
        return ""

    value_flags = {
        "--maxiterate", "--op", "--ep", "--bl", "--jt", "--thread",
        "-perm", "-perturb", "-replicates", "-threads", "-minconf", "-maxgapfract",
    }
    kept = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        base = tok.split("=")[0]
        if base in allowed:
            kept.append(tok)
            if base in value_flags and "=" not in tok:
                if i + 1 < len(tokens) and not tokens[i + 1].startswith("-"):
                    kept.append(tokens[i + 1])
                    i += 2
                    continue
        i += 1
    return " ".join(kept)


def _validate_action(response: Any) -> dict:
    if not isinstance(response, dict):
        raise ValueError("Response is not a dictionary")

    if response.get("error") == "invalid_json":
        raise json.JSONDecodeError(response.get("raw_output", ""), "", 0)

    allowed_actions = {"accept", "refine", "trim", "change_tool", "change_strategy", "stop"}
    allowed_tools = {"mafft", "muscle", "esm2"}
    allowed_modes = {"auto", "localpair", "globalpair", "fast", "align", "progressive", "super5", "local", "global"}
    allowed_confidence = {"high", "medium", "low"}

    action = response.get("action") or response.get("decision", "")
    tool = response.get("tool", "mafft")
    msa_mode = response.get("msa_mode", "auto")
    extra_args = response.get("extra_args", "")
    reason = response.get("reason", "")
    confidence = response.get("confidence", "medium")

    if not isinstance(action, str):
        action = "refine"
    elif action not in allowed_actions:
        action = "refine"

    if not isinstance(tool, str):
        tool = "mafft"
    elif tool not in allowed_tools:
        tool = "mafft"

    if not isinstance(msa_mode, str):
        msa_mode = "auto"
    elif msa_mode not in allowed_modes:
        msa_mode = "auto"

    if not isinstance(confidence, str):
        confidence = "medium"
    elif confidence not in allowed_confidence:
        confidence = "medium"

    if not isinstance(reason, str):
        reason = "No reason provided"
    if not isinstance(extra_args, str):
        extra_args = ""

    extra_args = _sanitize_extra_args(action, tool, extra_args)

    return {
        "action": action,
        "tool": tool,
        "msa_mode": msa_mode,
        "extra_args": extra_args,
        "reason": reason,
        "confidence": confidence,
        "metrics_echo": response.get("metrics_echo", {})
    }


def _check_metrics_echo(response: dict, actual_metrics: dict) -> bool:
    echo = response.get("metrics_echo")
    if not isinstance(echo, dict):
        return True

    for key in ("gap_fraction", "sp_score", "identity"):
        echoed = echo.get(key)
        actual = actual_metrics.get(key)
        if echoed is None or actual is None:
            continue
        try:
            if abs(float(echoed) - float(actual)) > 0.05:
                return False
        except (ValueError, TypeError):
            return False

    return True


def decide_action(metrics: dict, convergence: dict, extra_flags: Optional[dict] = None) -> dict:
    if extra_flags is None:
        extra_flags = {}

    prompt = _build_prompt(metrics, convergence, extra_flags)

    for attempt in range(3):
        try:
            result = ask_llm_json(prompt)
            validated = _validate_action(result)
            if not _check_metrics_echo(validated, metrics):
                if attempt == 2:
                    validated["reason"] = f"[metrics mismatch ignored after retries] {validated.get('reason', '')}"
                    validated["confidence"] = "low"
                else:
                    prompt = _build_retry_prompt(prompt, result)
                    continue
            return validated
        except (json.JSONDecodeError, ValueError) as e:
            if attempt == 2:
                raise ValueError(f"Failed to get valid LLM response after retries: {e}")
            prompt = _build_retry_prompt(prompt, result)

    raise ValueError("Unexpected error in LLM interaction")