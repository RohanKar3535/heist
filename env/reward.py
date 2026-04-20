"""
Reward Calculator — Step 5.

Three reward functions:
    R_investigator: 7-component weighted reward for the trained detective
    R_criminal:     multiplicative reward for the adversarial designer
    R_oversight:    anomaly-detection ratio for the Fleet AI oversight agent

Plus:
    Information-theoretic query scoring (entropy-based info gain)
    Shapley value calculator for multi-agent credit attribution (D6)
"""

from __future__ import annotations

import math
from itertools import combinations
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np

try:
    from .transaction_graph import TransactionGraph, OFFSHORE_JURISDICTIONS
except ImportError:
    from transaction_graph import TransactionGraph, OFFSHORE_JURISDICTIONS


# ---------------------------------------------------------------------------
# Default weights (tuned for GRPO signal quality)
# ---------------------------------------------------------------------------

DEFAULT_WEIGHTS: Dict[str, float] = {
    "w1_detection_f1":            0.50,
    "w2_evidence_quality":        0.30,
    "w3_query_efficiency":        0.05,
    "w4_jurisdiction_compliance": 0.05,
    "w5_false_positive_penalty":  0.05,
    "w6_novel_scheme_bonus":      0.025,
    "w7_missed_novel_penalty":    0.025,
}

# ---------------------------------------------------------------------------
# Helper: binary entropy
# ---------------------------------------------------------------------------

def _entropy(p: float) -> float:
    """Binary entropy  H(p) = -p*log(p) - (1-p)*log(1-p).  Clipped to [1e-9, 1-1e-9]."""
    p = max(1e-9, min(1.0 - 1e-9, p))
    return -p * math.log2(p) - (1.0 - p) * math.log2(1.0 - p)


# ---------------------------------------------------------------------------
# 1. R_investigator
# ---------------------------------------------------------------------------

def compute_detection_f1(
    evidence_chain: List[str],
    ground_truth_path: List[str],
) -> Tuple[float, float, float]:
    """
    F1 score between investigator's evidence_chain and ground truth scheme path.

    Returns (precision, recall, f1).
    """
    evidence = set(evidence_chain)
    truth    = set(ground_truth_path)

    if not truth:
        return (1.0, 1.0, 1.0) if not evidence else (0.0, 1.0, 0.0)

    overlap   = evidence & truth
    precision = len(overlap) / max(len(evidence), 1)
    recall    = len(overlap) / max(len(truth), 1)
    f1        = (2.0 * precision * recall) / max(precision + recall, 1e-9)
    return (round(precision, 6), round(recall, 6), round(f1, 6))


def compute_evidence_quality(compliance_score: float) -> float:
    """
    Evidence quality component — compliance score from the SAR filing tool.
    Comes from file_SAR().data.compliance_score (0–1 F1-based metric).
    """
    return float(max(0.0, min(1.0, compliance_score)))


def compute_query_efficiency(queries_used: int, total_budget: int = 50) -> float:
    """
    Query efficiency: -log2(queries_used / budget).
    More queries used → lower score.  Fewer queries → higher score.
    Returns 0 if all budget consumed.  Normalized to [0, 1].
    """
    if total_budget <= 0 or queries_used <= 0:
        return 1.0
    ratio = min(queries_used / total_budget, 1.0)
    # -log2(ratio) ranges from 0 (ratio=1) to ~5.6 (ratio=1/50)
    # Normalize to [0, 1] by dividing by log2(budget)
    raw = -math.log2(max(ratio, 1e-9))
    max_val = math.log2(total_budget)  # best case: 1 query out of 50
    return round(min(raw / max(max_val, 1e-9), 1.0), 6)


def compute_jurisdiction_compliance(
    query_history: List[Dict[str, Any]],
    ground_truth_path: List[str],
    graph: TransactionGraph,
) -> float:
    """
    Fraction of required jurisdictions that were cross-referenced.

    Required jurisdictions = unique jurisdictions of all ground-truth entities.
    Cross-referenced = jurisdictions queried via cross_reference_jurisdiction tool.
    """
    if not ground_truth_path:
        return 1.0

    # Collect required jurisdictions from GT path
    required_jurs: Set[str] = set()
    for eid in ground_truth_path:
        if graph.graph.has_node(eid):
            jur = graph.graph.nodes[eid].get("jurisdiction")
            if jur:
                required_jurs.add(jur)

    if not required_jurs:
        return 1.0

    # Collect cross-referenced jurisdictions from query history
    cross_ref_jurs: Set[str] = set()
    for entry in query_history:
        if entry.get("action_type") == "cross_reference_jurisdiction":
            eid = entry.get("params", {}).get("entity_id", "")
            if graph.graph.has_node(eid):
                jur = graph.graph.nodes[eid].get("jurisdiction")
                if jur:
                    cross_ref_jurs.add(jur)

    covered = len(required_jurs & cross_ref_jurs)
    return round(covered / len(required_jurs), 6)


def compute_false_positive_penalty(
    evidence_chain: List[str],
    ground_truth_path: List[str],
) -> float:
    """
    Fraction of evidence_chain entities that are NOT in the ground truth.
    Higher = more false positives = worse.
    """
    if not evidence_chain:
        return 0.0
    evidence = set(evidence_chain)
    truth    = set(ground_truth_path)
    false_pos = evidence - truth
    return round(len(false_pos) / len(evidence), 6)


def compute_novel_scheme_bonus(
    scheme_type: str,
    is_codex_generated: bool = False,
    detection_f1: float = 0.0,
) -> float:
    """
    Extra reward for successfully detecting a Criminal Codex-generated scheme.
    Returns 0.0 for seed schemes, bonus proportional to F1 for novel schemes.
    """
    if not is_codex_generated:
        return 0.0
    return round(detection_f1 * 1.5, 6)  # 50% multiplier for novel scheme detection


def compute_missed_novel_penalty(
    scheme_type: str,
    is_codex_generated: bool = False,
    detection_f1: float = 0.0,
) -> float:
    """
    Extra penalty for failing to detect a Criminal Codex-generated scheme.
    Returns 0.0 for seed schemes, penalty proportional to (1-F1) for novel schemes.
    """
    if not is_codex_generated:
        return 0.0
    return round((1.0 - detection_f1) * 1.5, 6)


def r_investigator(
    evidence_chain: List[str],
    ground_truth_path: List[str],
    compliance_score: float,
    queries_used: int,
    total_budget: int,
    query_history: List[Dict[str, Any]],
    graph: TransactionGraph,
    scheme_type: str = "",
    is_codex_generated: bool = False,
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Full investigator reward.

    R = w1*detection_f1
      + w2*evidence_quality
      + w3*query_efficiency
      + w4*jurisdiction_compliance
      - w5*false_positive_penalty
      + w6*novel_scheme_bonus
      - w7*missed_novel_penalty

    Returns dict with total reward and all components for interpretability.
    """
    w = {**DEFAULT_WEIGHTS, **(weights or {})}

    prec, rec, f1 = compute_detection_f1(evidence_chain, ground_truth_path)
    eq  = compute_evidence_quality(compliance_score)
    qe  = compute_query_efficiency(queries_used, total_budget)
    jc  = compute_jurisdiction_compliance(query_history, ground_truth_path, graph)
    fpp = compute_false_positive_penalty(evidence_chain, ground_truth_path)
    nsb = compute_novel_scheme_bonus(scheme_type, is_codex_generated, f1)
    mnp = compute_missed_novel_penalty(scheme_type, is_codex_generated, f1)

    total = (
        w["w1_detection_f1"]            * f1
        + w["w2_evidence_quality"]      * eq
        + w["w3_query_efficiency"]      * qe
        + w["w4_jurisdiction_compliance"] * jc
        - w["w5_false_positive_penalty"] * fpp
        + w["w6_novel_scheme_bonus"]    * nsb
        - w["w7_missed_novel_penalty"]  * mnp
    )

    return {
        "total":                    round(total, 6),
        "detection_f1":             f1,
        "precision":                prec,
        "recall":                   rec,
        "evidence_quality":         round(eq, 6),
        "query_efficiency":         round(qe, 6),
        "jurisdiction_compliance":  round(jc, 6),
        "false_positive_penalty":   round(fpp, 6),
        "novel_scheme_bonus":       round(nsb, 6),
        "missed_novel_penalty":     round(mnp, 6),
        "weights":                  w,
    }


# ---------------------------------------------------------------------------
# 2. R_criminal
# ---------------------------------------------------------------------------

def r_criminal(
    laundering_volume: float,
    detection_rate: float,
    novelty_bonus: float = 1.0,
    morph_success: bool = False,
) -> Dict[str, Any]:
    """
    Criminal reward: R = volume × (1 - detection_rate) × novelty × morph_bonus.

    Args:
        laundering_volume: total amount successfully laundered (normalized 0-1)
        detection_rate:    fraction of scheme detected by investigator (0-1)
        novelty_bonus:     multiplier for novel schemes (>1 for Codex-generated)
        morph_success:     True if morph partially evaded detection

    Returns dict with total and components.
    """
    laundering_volume = max(0.0, min(1.0, laundering_volume))
    detection_rate    = max(0.0, min(1.0, detection_rate))
    novelty_bonus     = max(1.0, novelty_bonus)
    morph_bonus       = 1.2 if morph_success else 1.0

    total = laundering_volume * (1.0 - detection_rate) * novelty_bonus * morph_bonus

    return {
        "total":              round(total, 6),
        "laundering_volume":  round(laundering_volume, 6),
        "detection_rate":     round(detection_rate, 6),
        "novelty_bonus":      round(novelty_bonus, 6),
        "morph_success_bonus": round(morph_bonus, 6),
    }


# ---------------------------------------------------------------------------
# 3. R_oversight
# ---------------------------------------------------------------------------

def r_oversight(
    anomalies_caught: int,
    anomalies_total: int,
    false_positives: int = 0,
) -> Dict[str, Any]:
    """
    Oversight agent reward: F1 over anomaly detection to prevent trivial all-flag policy.

    Args:
        anomalies_caught: true positives (anomalies correctly flagged)
        anomalies_total:  total anomalies present (for recall)
        false_positives:  benign transactions incorrectly flagged

    Returns dict with total (F1) and components.
    """
    anomalies_caught = max(0, anomalies_caught)
    false_positives  = max(0, false_positives)

    if anomalies_total <= 0:
        return {"total": 0.0, "precision": 0.0, "recall": 0.0,
                "anomalies_caught": anomalies_caught, "anomalies_total": anomalies_total,
                "false_positives": false_positives}

    recall    = min(anomalies_caught / anomalies_total, 1.0)
    predicted = anomalies_caught + false_positives
    precision = anomalies_caught / max(predicted, 1)
    f1 = (2.0 * precision * recall) / max(precision + recall, 1e-9)

    return {
        "total":            round(f1, 6),
        "precision":        round(precision, 6),
        "recall":           round(recall, 6),
        "anomalies_caught": anomalies_caught,
        "anomalies_total":  anomalies_total,
        "false_positives":  false_positives,
    }


# ---------------------------------------------------------------------------
# 4. Information-theoretic query scoring
# ---------------------------------------------------------------------------

def info_gain(p_before: float, p_after: float) -> float:
    """
    Information gain of a query:
        IG = H(P_before) - H(P_after)

    Positive = query reduced uncertainty about entity criminality.
    """
    return round(_entropy(p_before) - _entropy(p_after), 6)


def rank_queries_by_info_gain(
    beliefs_before: Dict[str, float],
    beliefs_after: Dict[str, float],
) -> List[Tuple[str, float]]:
    """
    Rank entities by information gain of the last query.
    Returns sorted list of (entity_id, info_gain) descending.
    """
    gains: List[Tuple[str, float]] = []
    for eid in beliefs_after:
        p_before = beliefs_before.get(eid, 0.1)
        p_after  = beliefs_after[eid]
        ig = info_gain(p_before, p_after)
        gains.append((eid, ig))
    gains.sort(key=lambda x: x[1], reverse=True)
    return gains


def expected_info_gain(
    beliefs: Dict[str, float],
    candidate_entity: str,
    p_suspicious: float = 0.5,
) -> float:
    """
    Expected information gain for querying a candidate entity.

    EIG = p_suspicious * IG(p→p_high) + (1-p_suspicious) * IG(p→p_low)

    Used for information-theoretic query selection (mutual information maximization).
    """
    prior = beliefs.get(candidate_entity, 0.1)

    # Hypothetical posteriors if query returns suspicious vs clean result
    # Using simplified likelihood ratios from tools.py
    lr_suspicious = 3.5  # high suspicion weight signal
    lr_clean      = 0.75  # normal signal

    prior_c = max(1e-9, min(1.0 - 1e-9, prior))

    p_high = (prior_c * lr_suspicious) / (prior_c * lr_suspicious + (1.0 - prior_c))
    p_low  = (prior_c * lr_clean) / (prior_c * lr_clean + (1.0 - prior_c))

    ig_high = info_gain(prior, p_high)
    ig_low  = info_gain(prior, p_low)

    eig = p_suspicious * ig_high + (1.0 - p_suspicious) * ig_low
    return round(eig, 6)


def select_best_query_target(
    beliefs: Dict[str, float],
    candidates: List[str],
) -> Optional[str]:
    """
    Select entity with highest expected information gain for next query.
    Implements mutual information maximization (PLAN.md Advanced Math #3).
    """
    if not candidates:
        return None

    # Handle empty beliefs: use default prior
    if not beliefs:
        beliefs = {c: 0.1 for c in candidates}

    best_eid: Optional[str] = None
    best_eig = -float("inf")

    for eid in candidates:
        eig = expected_info_gain(beliefs, eid)
        if eig > best_eig:
            best_eig = eig
            best_eid = eid

    return best_eid


# ---------------------------------------------------------------------------
# 5. Shapley value calculator
# ---------------------------------------------------------------------------

def shapley_values(
    agents: List[str],
    value_function: Callable[[frozenset], float],
) -> Dict[str, float]:
    """
    Exact Shapley value computation for multi-agent credit attribution.

    phi_i = sum over S ⊆ N\\{i}: [|S|!(N-|S|-1)!/N!] * [v(S∪{i}) - v(S)]

    Args:
        agents:          list of agent identifiers  (e.g. ["investigator", "expert", "oversight"])
        value_function:  v(S) → float, coalition value for any subset S

    Returns dict mapping agent → Shapley value.
    """
    n = len(agents)
    if n == 0:
        return {}

    factorial_n = math.factorial(n)
    phi: Dict[str, float] = {a: 0.0 for a in agents}

    for i, agent_i in enumerate(agents):
        others = [a for a in agents if a != agent_i]

        # Iterate over all subsets S of others
        for k in range(len(others) + 1):
            for combo in combinations(others, k):
                s = frozenset(combo)
                s_with_i = s | {agent_i}

                coeff = math.factorial(len(s)) * math.factorial(n - len(s) - 1) / factorial_n
                marginal = value_function(s_with_i) - value_function(s)
                phi[agent_i] += coeff * marginal

    # Round for cleanliness
    return {a: round(v, 6) for a, v in phi.items()}


def heist_shapley(
    investigator_f1: float,
    expert_compliance: float,
    oversight_ratio: float,
) -> Dict[str, float]:
    """
    Shapley values for the HEIST 3-agent coalition.

    Value function v(S):
        - investigator alone: detection_f1
        - expert alone: compliance score * 0.3 (limited without investigation)
        - oversight alone: oversight_ratio * 0.2 (limited without investigation)
        - investigator + expert: f1 * 0.7 + compliance * 0.3
        - investigator + oversight: f1 * 0.8 + oversight * 0.2
        - expert + oversight: compliance * 0.3 + oversight * 0.2 (no detection)
        - all three: f1 * 0.5 + compliance * 0.3 + oversight * 0.2
    """
    agents = ["investigator", "expert", "oversight"]

    def v(s: frozenset) -> float:
        has_inv = "investigator" in s
        has_exp = "expert" in s
        has_ov  = "oversight" in s

        val = 0.0
        if has_inv:
            val += investigator_f1 * (0.5 if (has_exp or has_ov) else 1.0)
        if has_exp:
            val += expert_compliance * 0.3
        if has_ov:
            val += oversight_ratio * 0.2
        return val

    return shapley_values(agents, v)


# ---------------------------------------------------------------------------
# Convenience: compute all rewards from episode state
# ---------------------------------------------------------------------------

def compute_episode_rewards(
    evidence_chain: List[str],
    graph: TransactionGraph,
    scheme_id: str,
    compliance_score: float,
    queries_used: int,
    total_budget: int,
    query_history: List[Dict[str, Any]],
    morph_occurred: bool = False,
    is_codex_generated: bool = False,
    anomalies_caught: int = 0,
    anomalies_total: int = 0,
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Compute all three rewards from a completed episode.

    Returns dict with investigator, criminal, oversight reward dicts + shapley.
    """
    gt = graph.ground_truth.get(scheme_id, {})
    ground_truth_path = gt.get("full_path", [])
    scheme_type       = gt.get("scheme_type", "")

    # Detection rate = recall
    _, rec, f1 = compute_detection_f1(evidence_chain, ground_truth_path)
    detection_rate = rec

    # Laundering volume: normalized by number of scheme edges
    injected = graph._injected.get(scheme_id, {})
    scheme_edges = injected.get("edges", [])
    if scheme_edges:
        total_amount = sum(
            float(graph.graph[e["src"]][e["tgt"]].get("amount", 0))
            for e in scheme_edges
            if graph.graph.has_edge(e["src"], e["tgt"])
        )
        # Normalize: smurfing ~$90k, layering ~$500k, trade ~$5M
        laundering_volume = min(total_amount / 5_000_000.0, 1.0)
    else:
        laundering_volume = 0.5  # default if scheme already cleared

    # Novelty bonus for Codex schemes
    novelty_bonus = 1.5 if is_codex_generated else 1.0

    # Compute rewards
    inv = r_investigator(
        evidence_chain=evidence_chain,
        ground_truth_path=ground_truth_path,
        compliance_score=compliance_score,
        queries_used=queries_used,
        total_budget=total_budget,
        query_history=query_history,
        graph=graph,
        scheme_type=scheme_type,
        is_codex_generated=is_codex_generated,
        weights=weights,
    )

    # Morph success: morph happened AND investigator F1 dropped below 0.6
    morph_success = morph_occurred and f1 < 0.6

    crim = r_criminal(
        laundering_volume=laundering_volume,
        detection_rate=detection_rate,
        novelty_bonus=novelty_bonus,
        morph_success=morph_success,
    )

    ov = r_oversight(
        anomalies_caught=anomalies_caught,
        anomalies_total=anomalies_total,
    )

    # Shapley attribution
    shap = heist_shapley(
        investigator_f1=f1,
        expert_compliance=compliance_score,
        oversight_ratio=ov["total"],
    )

    return {
        "investigator": inv,
        "criminal":     crim,
        "oversight":    ov,
        "shapley":      shap,
    }
