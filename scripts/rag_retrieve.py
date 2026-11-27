import gzip
import json
import os
import subprocess
from typing import Optional, List, Dict
from collections import defaultdict
import pathlib

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
KNOWLEDGE_DIR = pathlib.Path(os.environ.get("MSA_KNOWLEDGE_DIR") or (SCRIPT_DIR.parent / "knowledge"))
TOON_FILE = "experiments.toon"
TOON_GZ_FILE = "experiments.toon.gz"
JSONL_GZ_FILE = "experiments.jsonl.gz"
JSONL_FILE = "experiments.jsonl"
STRATEGY_STATS_JSON = "strategy_stats.json"
STRATEGY_STATS_TOON = "strategy_stats.toon"
STRATEGY_STATS_GZ = "strategy_stats.toon.gz"
ESM_GAP_STATS_GZ = "esm_gap_stats.toon.gz"
ESM_POST_STATS_GZ = "esm_post_stats.toon.gz"


def _gzip_read_text(path) -> str:
    out = subprocess.run(["gzip", "-dc", str(path)], capture_output=True, check=True)
    return out.stdout.decode("utf-8")


def _gzip_write_text(path, text: str) -> None:
    data = text.encode("utf-8")
    with open(path, "wb") as f:
        subprocess.run(["gzip", "-c"], input=data, stdout=f, check=True)


_CACHE_KNOWLEDGE: Optional[List[Dict]] = None
_CACHE_KNOWLEDGE_MTIME: float = 0.0


def load_knowledge() -> List[Dict]:
    global _CACHE_KNOWLEDGE, _CACHE_KNOWLEDGE_MTIME

    jsonl_gz_path = KNOWLEDGE_DIR / JSONL_GZ_FILE
    jsonl_path = KNOWLEDGE_DIR / JSONL_FILE
    toon_gz_path = KNOWLEDGE_DIR / TOON_GZ_FILE
    toon_path = KNOWLEDGE_DIR / TOON_FILE

    for data_path in [jsonl_gz_path, jsonl_path, toon_gz_path, toon_path]:
        if not data_path.exists():
            continue
        try:
            mtime = data_path.stat().st_mtime
            if _CACHE_KNOWLEDGE is not None and mtime == _CACHE_KNOWLEDGE_MTIME:
                return _CACHE_KNOWLEDGE
            result = []
            if data_path.suffix == ".gz":
                text = _gzip_read_text(data_path)
                if data_path.name.endswith(".jsonl.gz"):
                    for line in text.splitlines():
                        line = line.strip()
                        if line:
                            result.append(json.loads(line))
                else:
                    from toon_format import from_toon
                    data = from_toon(text)
                    if isinstance(data, list):
                        result = data
                    elif isinstance(data, dict):
                        for key in data:
                            if isinstance(data[key], list):
                                result = data[key]
                                break
            elif data_path.suffix == ".jsonl":
                with open(data_path, "r") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            result.append(json.loads(line))
            else:
                from toon_format import from_toon
                text = data_path.read_text()
                data = from_toon(text)
                if isinstance(data, list):
                    result = data
                elif isinstance(data, dict):
                    for key in data:
                        if isinstance(data[key], list):
                            result = data[key]
                            break
            _CACHE_KNOWLEDGE = result
            _CACHE_KNOWLEDGE_MTIME = mtime
            return result
        except Exception:
            pass

    jsonl_path = KNOWLEDGE_DIR / JSONL_FILE
    if jsonl_path.exists():
        experiments = []
        with open(jsonl_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        experiments.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return experiments

    return []


def _compute_sequence_signature(fasta_text: str, k: int = 5) -> set:
    seq = fasta_text.replace(">", "").replace("\n", "").replace(" ", "")
    return set(seq[i:i+k] for i in range(len(seq) - k + 1))


def _jaccard_similarity(set1: set, set2: set) -> float:
    if not set1 or not set2:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


def update_strategy_stats(experiment: Dict) -> None:
    from toon_format import to_toon, from_toon

    stats_path_gz = KNOWLEDGE_DIR / STRATEGY_STATS_GZ
    stats_path_toon = KNOWLEDGE_DIR / STRATEGY_STATS_TOON
    stats_path_json = KNOWLEDGE_DIR / STRATEGY_STATS_JSON

    stats = {}
    if stats_path_gz.exists():
        try:
            data = from_toon(_gzip_read_text(stats_path_gz))
            if isinstance(data, dict) and "strategy_stats" in data:
                stats = data["strategy_stats"]
        except Exception:
            pass
    elif stats_path_toon.exists():
        try:
            data = from_toon(stats_path_toon.read_text())
            if isinstance(data, dict) and "strategy_stats" in data:
                stats = data["strategy_stats"]
        except Exception:
            pass
    elif stats_path_json.exists():
        try:
            with open(stats_path_json, "r") as f:
                stats = json.load(f)
        except Exception:
            pass

    deduped = {}
    for k, v in stats.items():
        stripped = k.strip('"')
        if stripped in deduped:
            for field in ("success", "failure", "total"):
                deduped[stripped][field] += v.get(field, 0)
        else:
            deduped[stripped] = dict(v)
    stats = deduped

    actions = experiment.get("actions_tried", [])
    for action in actions:
        action_type = action.get("action", "unknown")
        strategy = action.get("strategy", "auto")
        extra = action.get("extra_args", "")
        outcome = action.get("outcome", "unknown")

        key = f"{action_type}:{strategy}:{extra}"

        if key not in stats:
            stats[key] = {"success": 0, "failure": 0, "total": 0}

        stats[key]["total"] += 1
        if outcome in ("success", "partial_improvement"):
            stats[key]["success"] += 1
        else:
            stats[key]["failure"] += 1

    wrapped = {"strategy_stats": stats}
    _gzip_write_text(stats_path_gz, to_toon(wrapped))


def get_strategy_recommendations(problem_type: str, gap_fraction: float) -> Dict[str, float]:
    from toon_format import from_toon

    stats_path_gz = KNOWLEDGE_DIR / STRATEGY_STATS_GZ
    stats_path_toon = KNOWLEDGE_DIR / STRATEGY_STATS_TOON
    stats_path_json = KNOWLEDGE_DIR / STRATEGY_STATS_JSON

    stats = {}
    if stats_path_gz.exists():
        try:
            data = from_toon(_gzip_read_text(stats_path_gz))
            if isinstance(data, dict) and "strategy_stats" in data:
                stats = data["strategy_stats"]
        except Exception:
            pass
    elif stats_path_toon.exists():
        try:
            data = from_toon(stats_path_toon.read_text())
            if isinstance(data, dict) and "strategy_stats" in data:
                stats = data["strategy_stats"]
        except Exception:
            pass
    elif stats_path_json.exists():
        try:
            with open(stats_path_json, "r") as f:
                stats = json.load(f)
        except Exception:
            pass

    recommendations = {}
    for strategy, data in stats.items():
        clean_key = strategy.strip('"')
        success_rate = data["success"] / max(data["total"], 1)
        if data["total"] >= 1:
            recommendations[clean_key] = {
                "success_rate": round(success_rate, 2),
                "total_runs": data["total"],
                "successes": data["success"]
            }

    return recommendations


def esm_gap_band(gap: float) -> str:
    """0.05-wide band label for a gap fraction (e.g. 0.12 -> '0.10-0.15')."""
    lo = int(gap * 20) / 20.0
    hi = lo + 0.05
    return f"{lo:.2f}-{hi:.2f}"


def load_esm_gap_stats() -> Dict[str, Dict]:
    from toon_format import from_toon

    stats_path_gz = KNOWLEDGE_DIR / ESM_GAP_STATS_GZ
    if not stats_path_gz.exists():
        return {}
    try:
        data = from_toon(_gzip_read_text(stats_path_gz))
    except Exception:
        return {}
    if isinstance(data, dict) and "bands" in data:
        return data["bands"] or {}
    if isinstance(data, dict):
        return data
    return {}


def update_esm_gap_stats(experiment: Dict) -> None:
    """Record one point per esm2 action: gap band -> n/success/sum_delta.

    success = gap dropped by >= 0.02 OR final gap after esm < 0.15.
    """
    from toon_format import to_toon

    actions = experiment.get("actions_tried") or []
    esm_entries = []
    for i, a in enumerate(actions):
        if not isinstance(a, dict) or a.get("tool") != "esm2":
            continue
        gap_after = float(a.get("gap_fraction") or 0)
        if i > 0 and isinstance(actions[i - 1], dict):
            gap_before = float(actions[i - 1].get("gap_fraction") or 0)
        else:
            gap_before = float(experiment.get("initial_gap_fraction") or 0)
        esm_entries.append((gap_before, gap_after))

    if not esm_entries:
        return

    stats = load_esm_gap_stats()
    for gap_before, gap_after in esm_entries:
        band = esm_gap_band(gap_before)
        if band not in stats:
            stats[band] = {"n": 0, "success": 0, "sum_delta": 0.0}
        improved = (gap_before - gap_after) >= 0.02 or gap_after < 0.15
        stats[band]["n"] += 1
        stats[band]["success"] += 1 if improved else 0
        stats[band]["sum_delta"] = round(
            stats[band]["sum_delta"] + (gap_before - gap_after), 4
        )

    KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
    _gzip_write_text(KNOWLEDGE_DIR / ESM_GAP_STATS_GZ, to_toon({"bands": stats}))


def load_esm_post_stats() -> Dict[str, Dict]:
    """Post-alignment ESM refine effectiveness by gap band (pre-refine gap)."""
    from toon_format import from_toon

    stats_path_gz = KNOWLEDGE_DIR / ESM_POST_STATS_GZ
    if not stats_path_gz.exists():
        return {}
    try:
        data = from_toon(_gzip_read_text(stats_path_gz))
    except Exception:
        return {}
    if isinstance(data, dict) and "bands" in data:
        return data["bands"] or {}
    if isinstance(data, dict):
        return data
    return {}


def update_esm_post_stats(experiment: Dict) -> None:
    """Record one point when ESM post-correction was applied.

    success = gap dropped by >= 0.02 OR gap after refine < 0.15.
    """
    from toon_format import to_toon

    if not experiment.get("esm_post_applied"):
        return

    gap_before = float(experiment.get("esm_gap_before") or 0)
    gap_after = float(experiment.get("esm_gap_after") or experiment.get("final_gap_fraction") or 0)
    band = esm_gap_band(gap_before)

    stats = load_esm_post_stats()
    if band not in stats:
        stats[band] = {"n": 0, "success": 0, "sum_delta": 0.0}
    improved = (gap_before - gap_after) >= 0.02 or gap_after < 0.15
    stats[band]["n"] += 1
    stats[band]["success"] += 1 if improved else 0
    stats[band]["sum_delta"] = round(
        stats[band]["sum_delta"] + (gap_before - gap_after), 4
    )

    KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
    _gzip_write_text(KNOWLEDGE_DIR / ESM_POST_STATS_GZ, to_toon({"bands": stats}))


def get_esm_post_advice(gap: float) -> str:
    """use | avoid | explore for post-align ESM refine at this gap.

    n>=3 and >=50% success -> use; n>=3 and <50% -> avoid; else explore.
    """
    band = esm_gap_band(gap)
    d = load_esm_post_stats().get(band)
    if not d or int(d.get("n", 0)) < 3:
        return "explore"
    n = int(d["n"])
    succ = int(d.get("success", 0))
    return "use" if (succ / n) >= 0.5 else "avoid"


def get_esm_post_hint() -> str:
    """One-line learned post-align ESM guidance (empty if no trials yet)."""
    stats = load_esm_post_stats()
    if not stats:
        return ""
    n = sum(int(d.get("n", 0)) for d in stats.values())
    succ = sum(int(d.get("success", 0)) for d in stats.values())
    if n == 0:
        return ""
    rate = succ / n * 100
    use_bands = []
    avoid_bands = []
    for band, d in stats.items():
        bn = int(d.get("n", 0))
        if bn < 3:
            continue
        (use_bands if int(d.get("success", 0)) / bn >= 0.5 else avoid_bands).append(band)
    parts = [f"Learned ESM post-correction: n={n}, success {rate:.0f}%."]
    if use_bands:
        parts.append(f"Apply when gap in {', '.join(sorted(use_bands))}.")
    if avoid_bands:
        parts.append(f"Skip when gap in {', '.join(sorted(avoid_bands))}.")
    return " ".join(parts)


def get_esm_threshold(default: float = 0.25) -> tuple:
    """Lowest gap band where ESM has n>=3 and success rate >= 50%.

    Bands with gap < 0.10 are ignored — already-good alignments cannot
    teach when ESM helps (they always "succeed" and would force thr=0).

    Returns (threshold, is_learned). Before enough data: (default, False).
    """
    stats = load_esm_gap_stats()
    min_band_lo = 0.10
    candidates = []
    for band, d in stats.items():
        n = int(d.get("n", 0))
        succ = int(d.get("success", 0))
        if n >= 3 and succ / n >= 0.5:
            try:
                lo = float(band.split("-")[0])
            except (ValueError, IndexError):
                continue
            if lo < min_band_lo:
                continue
            candidates.append(lo)
    if candidates:
        return min(candidates), True
    return default, False


def get_esm_gap_hint() -> str:
    """One-line learned ESM guidance for the LLM prompt (empty if not learned)."""
    thr, learned = get_esm_threshold()
    if not learned:
        return ""
    stats = load_esm_gap_stats()
    n = sum(int(d.get("n", 0)) for d in stats.values())
    succ = sum(int(d.get("success", 0)) for d in stats.values())
    rate = (succ / n * 100) if n else 0
    return (
        f"Learned: ESM-2 works when gap > {thr:.2f} "
        f"(esm trials n={n}, success {rate:.0f}%). Prefer change_tool esm2 above this gap."
    )


def _sparse_search(experiments: List[Dict], fasta_text: str, exclude_id: Optional[str], top_k: int) -> List[Dict]:
    from tfidf_sparse import load_index, compute_query_vector, query_sparse

    matrix, vocab, exp_ids = load_index()
    if matrix.shape[0] == 0 or not vocab:
        return []

    query_vec = compute_query_vector(fasta_text, vocab)
    results = query_sparse(query_vec, matrix, exp_ids, top_k=len(exp_ids))

    id_to_exp = {exp.get("experiment_id"): exp for exp in experiments}
    filtered = []
    for exp_id, sim in results:
        if exp_id == exclude_id:
            continue
        if exp_id in id_to_exp:
            exp = id_to_exp[exp_id].copy()
            exp["_similarity"] = round(sim, 4)
            filtered.append(exp)
            if len(filtered) >= top_k:
                break
    return filtered


WARM_START_THRESHOLD = 0.8


def _infer_tool(action: Dict) -> str:
    tool = action.get("tool")
    action_name = action.get("action", "")
    strategy = (action.get("strategy") or "").strip()
    extra = (action.get("extra_args") or "").strip()
    # trim steps always ran trimal (tool field may hold the aligner that fed it)
    if action_name == "trim" or action.get("trim_strategy"):
        return "trimal"
    if extra.startswith("-gappy") or extra in (
        "-strict", "-strictplus", "-automated1", "-nogaps"
    ):
        return "trimal"
    if tool:
        return tool
    if strategy in ("local", "global"):
        return "esm2"
    if strategy in ("align", "progressive", "super5"):
        return "muscle"
    return "mafft"


def derive_warm_plan(actions_tried: Optional[List[Dict]]) -> Dict:
    """Build start/plan/avoid from a prior experiment's action list.

    start: tool/strategy of lowest-gap successful align step (not trimal).
    plan: later successful actions (tool/strategy/extra).
    avoid: failed actions (tool/strategy/extra).
    """
    plan = {
        "start": None,
        "plan": [],
        "avoid": [],
        "esm_direct": False,
        "esm_post_won": False,
    }
    if not actions_tried:
        return plan

    best_align = None
    best_gap = None
    for a in actions_tried:
        if not isinstance(a, dict):
            continue
        tool = _infer_tool(a)
        outcome = a.get("outcome")
        is_ok = outcome in ("success", "partial_improvement")
        is_failed = outcome in ("failure", "no_improvement")
        gap = a.get("gap_fraction")
        gap_f = float(gap) if isinstance(gap, (int, float)) else 999.0
        if tool == "trimal":
            entry = {
                "tool": "trimal",
                "strategy": a.get("trim_strategy") or a.get("extra_args") or "",
                "extra_args": a.get("extra_args") or "",
                "gap_fraction": a.get("gap_fraction"),
            }
            if is_ok or outcome == "no_change":
                plan["plan"].append(entry)
            else:
                plan["avoid"].append(entry)
            continue
        if is_failed:
            plan["avoid"].append({
                "tool": tool,
                "strategy": a.get("strategy") or "",
                "extra_args": a.get("extra_args") or "",
                "gap_fraction": a.get("gap_fraction"),
            })
            continue
        # eligible for start: success, partial, or intermediate no_change
        if best_align is None or gap_f < best_gap:
            best_align = {
                "tool": tool,
                "strategy": a.get("strategy") or "",
                "extra_args": a.get("extra_args") or "",
                "gap_fraction": gap,
            }
            best_gap = gap_f

    if best_align:
        plan["start"] = best_align
        plan["esm_direct"] = best_align["tool"] == "esm2"
        seen_start = False
        for a in actions_tried:
            if not isinstance(a, dict):
                continue
            tool = _infer_tool(a)
            if tool == "trimal":
                continue
            outcome = a.get("outcome")
            if outcome in ("failure", "no_improvement"):
                continue
            if not seen_start:
                if (
                    tool == best_align["tool"]
                    and (a.get("strategy") or "") == best_align["strategy"]
                    and a.get("gap_fraction") == best_align.get("gap_fraction")
                ):
                    seen_start = True
                elif tool == best_align["tool"] and not seen_start:
                    # first match by tool is enough if gap matches loosely
                    if a.get("gap_fraction") == best_align.get("gap_fraction"):
                        seen_start = True
                continue
            plan["plan"].append({
                "tool": tool,
                "strategy": a.get("strategy") or "",
                "extra_args": a.get("extra_args") or "",
                "gap_fraction": a.get("gap_fraction"),
            })
    return plan


def find_warm_start(fasta_text: str, exclude_id: Optional[str] = None, threshold: float = WARM_START_THRESHOLD) -> Optional[Dict]:
    experiments = load_knowledge()
    if not experiments:
        return None

    sparse_results = _sparse_search(experiments, fasta_text, exclude_id, top_k=1)
    if not sparse_results:
        return None

    best = sparse_results[0]
    sim = best.get("_similarity", 0)
    if sim < threshold:
        return None

    actions = best.get("actions_tried", [])
    return {
        "experiment_id": best.get("experiment_id"),
        "similarity": sim,
        "actions_tried": actions,
        "final_outcome": best.get("final_outcome"),
        "final_gap_fraction": best.get("final_gap_fraction"),
        "problem_type": best.get("problem_type"),
        "plan": derive_warm_plan(actions),
        "esm_post_applied": bool(best.get("esm_post_applied")),
    }


def retrieve_relevant(
    problem_type: Optional[str] = None,
    gap_fraction: Optional[float] = None,
    exclude_id: Optional[str] = None,
    top_k: int = 3,
    fasta_text: Optional[str] = None
) -> List[Dict]:
    experiments = load_knowledge()

    if not experiments:
        return []

    if fasta_text:
        sparse_results = _sparse_search(experiments, fasta_text, exclude_id, top_k)
        if sparse_results:
            return sparse_results

    current_sig = _compute_sequence_signature(fasta_text) if fasta_text else None

    scored = []
    for exp in experiments:
        if exp.get("experiment_id") == exclude_id:
            continue

        score = 0

        if problem_type and exp.get("problem_type") == problem_type:
            score += 3

        if gap_fraction is not None:
            exp_gap = exp.get("initial_gap_fraction", 0)
            if gap_fraction > 0.2 and exp_gap > 0.2:
                score += 2
            elif abs(gap_fraction - exp_gap) < 0.1:
                score += 1

        if exp.get("final_outcome") == "success":
            score += 2
        elif exp.get("final_outcome") == "partial_improvement":
            score += 1

        if current_sig and exp.get("sequence_signature"):
            exp_sig = set(exp["sequence_signature"]) if isinstance(exp["sequence_signature"], list) else exp["sequence_signature"]
            sim = _jaccard_similarity(current_sig, exp_sig)
            score += int(sim * 3)

        if score > 0:
            scored.append((score, exp))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [exp for _, exp in scored[:top_k]]


def get_failure_lessons(problem_type: str) -> List[str]:
    experiments = load_knowledge()

    failures = []
    for exp in experiments:
        if exp.get("problem_type") != problem_type:
            continue
        if exp.get("final_outcome") == "no_improvement":
            actions = exp.get("actions_tried", [])
            for action in actions:
                if action.get("outcome") == "no_change":
                    failures.append(
                        f"Don't use {action.get('action')} with {action.get('strategy')} {action.get('extra_args')}"
                    )

    return failures[:3]


def format_retrieved(experiments: List[Dict], problem_type: str = None) -> str:
    from toon_format import to_toon

    if not experiments:
        return "No previous experiments found in knowledge base."

    summary_records = []
    for exp in experiments:
        summary_records.append({
            "problem_type": exp.get("problem_type"),
            "initial_gap": exp.get("initial_gap_fraction"),
            "final_gap": exp.get("final_gap_fraction"),
            "outcome": exp.get("final_outcome"),
            "similarity": exp.get("_similarity"),
            "lesson": exp.get("lesson", "")
        })

    context = "=== PREVIOUS RELEVANT EXPERIMENTS (TOON format) ===\n"
    context += to_toon({"relevant_experiments": summary_records})

    if problem_type:
        failures = get_failure_lessons(problem_type)
        if failures:
            context += "\n=== WHAT NOT TO DO ===\n"
            for f in failures:
                context += f"  - {f}\n"

    recommendations = get_strategy_recommendations(problem_type or "high_gap_fraction", 0.3)
    if recommendations:
        context += "\n=== SUCCESSFUL STRATEGIES ===\n"
        sorted_recs = sorted(recommendations.items(), key=lambda x: x[1]["success_rate"], reverse=True)
        for strategy, data in sorted_recs[:3]:
            context += f"  - {strategy}: {data['success_rate']*100}% success ({data['total_runs']} runs)\n"

    return context