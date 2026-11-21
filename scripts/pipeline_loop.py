import uuid
from typing import Optional, List, Dict
from datetime import datetime

from msa_runner import run_msa
from muscle_runner import run_muscle
from trimal_runner import run_trimal
from esm_aligner import run_esm_alignment
from extended_metrics import compute_extended_metrics
from convergence import ConvergenceTracker
from ai_controller import decide_action
from logger import ExperimentLogger
from rag_store import store_experiment
from rag_retrieve import (
    find_warm_start, get_esm_threshold, get_esm_gap_hint,
    get_esm_post_advice, get_esm_post_hint,
)
from rag_context import build_rag_context
from rag_advanced import get_weighted_recommendations, load_experiments as rag_load_experiments
from knowledge_base import get_external_context


DEFAULT_FASTA = """>seq1
ATCGATCGATCG
>seq2
ATCGATCGATCG
>seq3
ATCGATCGAT--
"""

MAFFT_STRATEGIES = ["--auto", "--localpair", "--globalpair", "--genafpair", "--retree 1"]

# Gentle only: strict/strictplus/nogaps delete gap columns (lose indels).
TRIM_STRATEGY_ORDER = ["-gappyout", "-automated1"]
VALID_TRIM_STRATEGIES = set(TRIM_STRATEGY_ORDER)

ESM_PROBE_GAP_MIN = 0.12
ESM_PROBE_GAP_MAX = 0.40
ESM_POST_BAD_GAP = 0.12


def _pick_trim_strategy(gap: float, tried: list, llm_choice: str = "") -> str:
    if llm_choice in VALID_TRIM_STRATEGIES and llm_choice not in tried:
        tried.append(llm_choice)
        return llm_choice

    # Always start gentle — never strip all gap/indel columns.
    severity = "-gappyout"
    for strategy in [severity] + [s for s in TRIM_STRATEGY_ORDER if s != severity]:
        if strategy not in tried:
            tried.append(strategy)
            return strategy

    tried.append(severity)
    return severity


def run_alignment(fasta_text: str, params: dict) -> str:
    tool = params.get("tool", "mafft")
    if tool == "muscle":
        return run_muscle(fasta_text, params)
    if tool == "esm2":
        return run_esm_alignment(fasta_text, params)
    return run_msa(fasta_text, params)


def _rule_based_decision(
    metrics: dict,
    current_tool: str,
    tool_streak: int,
    esm_tried: bool = False,
) -> Optional[dict]:
    gap = metrics.get("gap_fraction", 0)
    sp = metrics.get("sp_score", 0)
    identity = metrics.get("identity", 0)

    if gap < 0.1 and sp > 0.5:
        return {"action": "accept", "tool": current_tool, "msa_mode": "auto",
                "extra_args": "", "reason": f"rule: gap={gap:.3f}<0.1, sp={sp:.3f}>0.5",
                "confidence": "high", "_rule_fired": True}

    # ESM probe / learned threshold: one esm2 try when gap qualifies
    if not esm_tried and current_tool != "esm2":
        thr, is_learned = get_esm_threshold()
        in_probe_band = ESM_PROBE_GAP_MIN <= gap <= ESM_PROBE_GAP_MAX
        above_learned = is_learned and gap > thr
        if in_probe_band:
            return {"action": "change_tool", "tool": "esm2", "msa_mode": "local",
                    "extra_args": "",
                    "reason": f"rule: esm probe gap={gap:.3f} in [{ESM_PROBE_GAP_MIN},{ESM_PROBE_GAP_MAX}], trying esm2",
                    "confidence": "medium", "_rule_fired": True}
        if above_learned:
            return {"action": "change_tool", "tool": "esm2", "msa_mode": "local",
                    "extra_args": "",
                    "reason": f"rule: learned esm threshold gap={gap:.3f}>{thr:.2f}, trying esm2",
                    "confidence": "medium", "_rule_fired": True}

    if gap > 0.25 and identity < 0.2 and current_tool != "esm2":
        return {"action": "change_tool", "tool": "esm2", "msa_mode": "local",
                "extra_args": "", "reason": f"rule: gap={gap:.3f}>0.25, identity={identity:.3f}<0.2",
                "confidence": "high", "_rule_fired": True}

    if gap > 0.15 and current_tool == "mafft":
        return {"action": "change_tool", "tool": "muscle", "msa_mode": "align",
                "extra_args": "", "reason": f"rule: gap={gap:.3f}>0.15 on mafft, switching to muscle",
                "confidence": "high", "_rule_fired": True}

    if gap > 0.15 and current_tool in ("muscle", "esm2") and tool_streak >= 2:
        return {"action": "trim", "tool": current_tool, "msa_mode": "auto",
                "extra_args": "-gappyout",
                "reason": f"rule: gap={gap:.3f}>0.15, {current_tool} tried {tool_streak}x, gentle trim",
                "confidence": "high", "_rule_fired": True}

    if identity < 0.3 and gap > 0.15 and current_tool != "muscle" and current_tool != "esm2":
        return {"action": "change_tool", "tool": "muscle", "msa_mode": "align",
                "extra_args": "", "reason": f"rule: identity={identity:.3f}<0.3, diverse sequences need muscle",
                "confidence": "high", "_rule_fired": True}

    return None


def _classify_problem(metrics: dict) -> str:
    gap = metrics.get("gap_fraction", 0)
    sp = metrics.get("sp_score", 0)
    identity = metrics.get("identity", 0)

    if gap > 0.2 and sp < 0.3:
        return "high_gap_fraction"
    if identity < 0.3:
        return "diverse_sequences"
    if gap > 0.15:
        return "moderate_gap_fraction"
    if gap < 0.1 and sp > 0.5:
        return "good_quality"
    return "general"


def run_pipeline(
    fasta_text: str = DEFAULT_FASTA,
    max_iterations: int = 10,
    convergence_tolerance: float = 0.001,
    convergence_patience: int = 3,
    params: Optional[dict] = None,
    warm_start_threshold: float = 0.8,
    esm_analysis: bool = False,
    esm_refine_gaps: bool = False,
    esm_alpha: float = 2.0,
    esm_post: str = "if_bad",
) -> dict:
    """esm_post: if_bad (default; gap>0.12 + not learned-avoid) | always | off.

    Learning uses if_bad so ESM post-correction only runs when required
    or learned-useful — not on every (good) family.
    """
    if params is None:
        params = {"tool": "mafft", "strategy": "--auto"}

    experiment_id = str(uuid.uuid4())[:8]
    logger = ExperimentLogger()
    tracker = ConvergenceTracker(tolerance=convergence_tolerance, patience=convergence_patience)

    current_fasta = fasta_text
    current_params = params.copy()
    was_trimmed = False
    just_trimmed = False
    metrics_history: List[Dict] = []
    decisions_history: List[Dict] = []
    last_gap = 1.0
    no_improvement_count = 0
    current_tool_streak = 0
    trim_tried: List[str] = []
    esm_tried = False

    warm_start = find_warm_start(fasta_text, experiment_id, warm_start_threshold)
    warm_start_context = ""
    warm_plan = None
    if warm_start:
        ws_id = warm_start["experiment_id"]
        ws_sim = warm_start["similarity"]
        ws_actions = warm_start["actions_tried"]
        warm_plan = warm_start.get("plan") or {}
        print(f"  Warm start: using actions from experiment {ws_id} (similarity: {ws_sim:.2f})")

        # Seed starting params from prior winner (may be esm2 = "direct esm")
        start = warm_plan.get("start")
        if start and start.get("tool"):
            current_params["tool"] = start["tool"]
            if start.get("strategy"):
                if start["tool"] == "mafft" and not str(start["strategy"]).startswith("--"):
                    current_params["strategy"] = f"--{start['strategy']}"
                else:
                    current_params["strategy"] = start["strategy"]
            if start.get("extra_args"):
                current_params["extra_args"] = start["extra_args"]
            if start["tool"] == "esm2":
                esm_tried = True
            print(f"  -> Seeded start: tool={current_params.get('tool')} strategy={current_params.get('strategy')}")

        for av in warm_plan.get("avoid") or []:
            if av.get("tool") == "trimal":
                flag = av.get("strategy") or av.get("extra_args") or ""
                if flag and flag not in trim_tried:
                    trim_tried.append(flag)

        warm_start_context = "\n=== WARM START: PREVIOUSLY TRIED ACTIONS ===\n"
        warm_start_context += f"Source experiment: {ws_id} (similarity: {ws_sim:.2f})\n"
        warm_start_context += f"Previous outcome: {warm_start['final_outcome']} (final gap: {warm_start['final_gap_fraction']})\n"
        if start:
            warm_start_context += (
                f"Seed start: tool={start.get('tool')} strategy={start.get('strategy')} "
                f"extra={start.get('extra_args', '')} (direct esm={warm_plan.get('esm_direct', False)})\n"
            )
        if warm_start.get("esm_post_applied"):
            warm_start_context += "Prior run used ESM post-correction successfully.\n"
        warm_start_context += "Actions tried:\n"
        for a in ws_actions:
            status = "OK" if a.get("outcome") in ("success", "partial_improvement") else "FAILED"
            tool = a.get("tool") or "?"
            warm_start_context += f"  - [{status}] tool={tool} {a.get('action')} {a.get('strategy', '')} {a.get('extra_args', '')} -> gap={a.get('gap_fraction', '?')}\n"
        warm_start_context += "Avoid repeating FAILED actions. Consider trying different strategies.\n"
        for av in warm_plan.get("avoid") or []:
            if av.get("tool") != "trimal":
                warm_start_context += f"  AVOID: tool={av.get('tool')} {av.get('strategy')} {av.get('extra_args')}\n"

    esm_post_hint = get_esm_post_hint()

    for iteration in range(1, max_iterations + 1):
        print(f"Iteration {iteration}/{max_iterations}")

        tool = current_params.get("tool", "mafft")
        print(f"  Tool: {tool}")
        aligned = run_alignment(current_fasta, current_params)
        metrics = compute_extended_metrics(aligned)

        converged, conv_info = tracker.check(metrics)

        extra_args = current_params.get("extra_args", "")

        problem_type = _classify_problem(metrics)

        rag_context = build_rag_context(
            gap_fraction=metrics.get("gap_fraction", 0),
            exclude_experiment_id=experiment_id,
            fasta_text=fasta_text
        )

        weighted_recs = get_weighted_recommendations(
            rag_load_experiments(),
            problem_type,
            datetime.now()
        )

        external_context = get_external_context(problem_type)

        combined_context = f"{rag_context}\n\n{external_context}\n\nWEIGHTED STRATEGY RECOMMENDATIONS:\n"
        for strategy, data in sorted(weighted_recs.items(), key=lambda x: x[1]["weighted_score"], reverse=True)[:5]:
            combined_context += f"  - {strategy}: score={data['weighted_score']}, success={data['success_rate']}\n"

        esm_hint = get_esm_gap_hint()
        if esm_hint:
            combined_context += f"\n{esm_hint}\n"
        if esm_post_hint:
            combined_context += f"\n{esm_post_hint}\n"

        final_context = f"{warm_start_context}\n{combined_context}" if warm_start_context else combined_context

        rule_decision = _rule_based_decision(metrics, tool, current_tool_streak, esm_tried=esm_tried)
        if rule_decision is not None:
            decision = rule_decision
            print(f"  [rule] {decision['reason']}")
        else:
            try:
                decision = decide_action(metrics, conv_info, {
                    "extra_args": extra_args,
                    "trimmed": was_trimmed,
                    "current_tool": tool,
                    "rag_context": final_context
                })
            except Exception as e:
                print(f"  [llm-fallback] LLM failed ({type(e).__name__}), using safe decision")
                gap = metrics.get("gap_fraction", 0)
                if gap < 0.15:
                    decision = {"action": "accept", "tool": tool, "msa_mode": "auto",
                                "extra_args": "", "reason": f"llm-fallback: gap={gap:.3f}<0.15",
                                "confidence": "low"}
                else:
                    decision = {"action": "trim", "tool": tool, "msa_mode": "auto",
                                "extra_args": "-gappyout",
                                "reason": f"llm-fallback: gap={gap:.3f}>0.15, gentle trim",
                                "confidence": "low"}

        print(f"  action: {decision['action']}, instability: {conv_info['instability']:.4f}, gap: {metrics['gap_fraction']:.3f}, reason: {decision.get('reason', '')[:60]}")

        # tool that produced this iteration's metrics (for KB actions_tried.tool)
        decision = {**decision, "_tool_used": tool}
        if decision.get("action") == "change_tool" and decision.get("tool") == "esm2":
            esm_tried = True
        if tool == "esm2":
            esm_tried = True

        metrics_history.append(metrics)
        decisions_history.append(decision)

        if metrics.get("gap_fraction", 0) >= last_gap - 0.005:
            no_improvement_count += 1
        else:
            no_improvement_count = 0
        last_gap = metrics.get("gap_fraction", 0)

        if tool == current_params.get("tool", "mafft"):
            current_tool_streak += 1
        else:
            current_tool_streak = 1

        is_bad_alignment = metrics.get("gap_fraction", 0) > 0.15

        if is_bad_alignment and current_tool_streak >= 2 and decision["action"] in ("refine", "change_strategy", "accept"):
            tool_cycle = ["mafft", "muscle", "esm2"]
            current_idx = tool_cycle.index(tool) if tool in tool_cycle else 0
            other_tool = tool_cycle[(current_idx + 1) % len(tool_cycle)]
            print(f"  -> Auto-switch: {tool} tried {current_tool_streak}x with gap={metrics['gap_fraction']:.3f}, forcing switch to {other_tool}")
            decision["action"] = "change_tool"
            decision["tool"] = other_tool
            if other_tool == "muscle":
                decision["msa_mode"] = "align"
            elif other_tool == "esm2":
                decision["msa_mode"] = "local"
                esm_tried = True
            else:
                decision["msa_mode"] = "auto"
            no_improvement_count = 0
            current_tool_streak = 0

        if decision["action"] == "accept" and is_bad_alignment:
            print(f"  -> Override: alignment still bad (gap={metrics['gap_fraction']:.3f}), continuing")
            decision["action"] = "refine"

        if converged and is_bad_alignment:
            print(f"  -> Override: converged but quality still bad (gap={metrics['gap_fraction']:.3f}), trying trim")
            decision["action"] = "trim"
            converged = False

        if decision["action"] == "trim":
            trim_strategy = _pick_trim_strategy(
                metrics.get("gap_fraction", 0), trim_tried, decision.get("extra_args", "")
            )
            decision["extra_args"] = trim_strategy
            print(f"  -> Trimming with: {trim_strategy}")
            trimal_params = {"strategy": trim_strategy}
            aligned = run_trimal(aligned, trimal_params)
            metrics = compute_extended_metrics(aligned)
            was_trimmed = True
            just_trimmed = True
            tracker = ConvergenceTracker(tolerance=convergence_tolerance, patience=convergence_patience)
            current_params["extra_args"] = ""
            print(f"  -> After trim: gap={metrics['gap_fraction']:.3f}, len={metrics['alignment_length']}")

            if metrics.get("gap_fraction", 1) < 0.1:
                print(f"  -> Trimmed alignment looks good! Accepting.")
                trim_tried = []
                logger.log_step({
                    "iteration": iteration,
                    "params": current_params,
                    "metrics": metrics,
                    "convergence": {"status": "trimmed_accepted"},
                    "decision": {"action": "accept", "reason": "Trimmed alignment is good"},
                    "trimmed": was_trimmed
                })
                break

            decision["action"] = "refine"
            decision["extra_args"] = ""

        if decision["action"] in ("accept", "stop") or converged:
            if decision["action"] == "accept":
                print("  -> Accepted")
            elif converged:
                print("  -> Converged")
            else:
                print("  -> Stopped")
            logger.log_step({
                "iteration": iteration,
                "params": current_params,
                "metrics": metrics,
                "convergence": conv_info,
                "decision": decision,
                "trimmed": was_trimmed
            })
            break

        if decision["action"] == "change_strategy":
            current_params["strategy"] = f"--{decision['msa_mode']}"
            print(f"  -> Strategy changed to {current_params['strategy']}")

        if decision["action"] == "change_tool":
            new_tool = decision.get("tool", "mafft")
            current_params["tool"] = new_tool
            current_params["extra_args"] = ""
            if new_tool == "muscle":
                current_params["strategy"] = decision.get("msa_mode", "align")
            elif new_tool == "esm2":
                current_params["strategy"] = decision.get("msa_mode", "local")
            else:
                current_params["strategy"] = f"--{decision.get('msa_mode', 'auto')}"
            print(f"  -> Switched to {new_tool} with {current_params['strategy']}")

        if decision["action"] == "refine":
            if current_params.get("tool", "mafft") == "mafft":
                current = current_params.get("strategy", "--auto")
                idx = MAFFT_STRATEGIES.index(current) if current in MAFFT_STRATEGIES else 0
                current_params["strategy"] = MAFFT_STRATEGIES[(idx + 1) % len(MAFFT_STRATEGIES)]
                print(f"  -> Refined: cycling to {current_params['strategy']}")

        if decision.get("extra_args") and not just_trimmed:
            current_params["extra_args"] = decision["extra_args"]
            print(f"  -> Extra args: {decision['extra_args']}")

        logger.log_step({
            "iteration": iteration,
            "params": current_params,
            "metrics": metrics,
            "convergence": conv_info,
            "decision": decision,
            "trimmed": was_trimmed
        })

        just_trimmed = False

    logger.save(experiment_id)
    print(f"Experiment saved: {experiment_id}")

    # ESM post-correction after main pipeline
    # always = learning; if_bad = only if still bad (and not learned-avoid); off = never
    esm_post_applied = False
    esm_gap_before = metrics.get("gap_fraction", 0) if metrics else 0
    esm_gap_after = esm_gap_before
    esm_result = None
    should_post = False
    post_reason = ""

    if esm_post == "always":
        should_post = True
        post_reason = "always"
    elif esm_post == "if_bad" and esm_gap_before > ESM_POST_BAD_GAP:
        advice = get_esm_post_advice(esm_gap_before)
        if advice != "avoid":
            should_post = True
            post_reason = f"required: gap={esm_gap_before:.3f}>{ESM_POST_BAD_GAP}, advice={advice}"
        else:
            post_reason = f"skip: advice=avoid gap={esm_gap_before:.3f}"
    elif esm_post == "off":
        post_reason = "esm_post=off"
    else:
        post_reason = f"gap={esm_gap_before:.3f}<={ESM_POST_BAD_GAP}"

    # legacy explicit flags still force analysis/refine
    run_refine = should_post or esm_refine_gaps
    run_analysis = should_post or esm_analysis

    if run_analysis:
        from esm_analysis import run_esm_analysis
        print(f"  ESM post-correction ({post_reason})")
        esm_result = run_esm_analysis(
            fasta_text=fasta_text,
            alignment=aligned,
            refine_gaps_flag=run_refine,
            alpha=esm_alpha
        )
        if esm_result.get("final_alignment"):
            aligned = esm_result["final_alignment"]
            if run_refine and should_post:
                esm_post_applied = True
                new_metrics = compute_extended_metrics(aligned)
                esm_gap_after = new_metrics.get("gap_fraction", esm_gap_before)
                metrics = new_metrics
                if metrics_history:
                    metrics_history[-1] = metrics
                print(f"  -> Post-refine gap: {esm_gap_before:.3f} -> {esm_gap_after:.3f}")

    if esm_post_applied:
        print(f"  ESM post applied: {esm_gap_before:.3f} -> {esm_gap_after:.3f} "
              f"(success={(esm_gap_before - esm_gap_after) >= 0.02 or esm_gap_after < 0.15})")

    store_experiment(
        experiment_id=experiment_id,
        fasta_text=fasta_text,
        metrics_history=metrics_history,
        decisions=decisions_history,
        final_metrics=metrics,
        was_trimmed=was_trimmed,
        final_alignment=aligned,
        esm_post_applied=esm_post_applied,
        esm_gap_before=esm_gap_before,
        esm_gap_after=esm_gap_after,
    )

    result = {
        "experiment_id": experiment_id,
        "iterations": iteration,
        "final_metrics": metrics,
        "final_alignment": aligned,
        "was_trimmed": was_trimmed,
        "esm_post_applied": esm_post_applied,
        "esm_gap_before": esm_gap_before,
        "esm_gap_after": esm_gap_after,
    }

    if esm_result:
        result["esm_analysis"] = esm_result.get("analysis", {})
        result["esm_report"] = esm_result.get("report", "")

    return result


if __name__ == "__main__":
    run_pipeline()