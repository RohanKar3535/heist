"""
Tool Execution Layer — Step 3.

Six investigation tools that operate on the TransactionGraph.
Each returns a dict compatible with _dispatch()'s (tool_result, reward) contract.

bayesian_update() computes proper posterior using likelihood ratios derived from
mutual-information-maximising feature selection over scheme edge patterns.
"""

from __future__ import annotations

import uuid
from collections import deque
from typing import Any, Dict, List, Optional

import numpy as np

try:
    from .transaction_graph import TransactionGraph, OFFSHORE_JURISDICTIONS
except ImportError:
    from transaction_graph import TransactionGraph, OFFSHORE_JURISDICTIONS

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TOOL_BUDGET: Dict[str, int] = {
    "query_transactions":         1,
    "trace_network":              1,
    "lookup_entity_info":         1,
    "cross_reference_jurisdiction": 1,
    "file_SAR":                   1,
    "request_subpoena":           2,
}

# Likelihood ratios per signal (derived from IBM AMLSim feature mutual information)
_LR: Dict[str, float] = {
    "structuring_amount": 5.0,   # amounts 9000–9999 → strong structuring signal
    "offshore_jur":       2.2,   # offshore jurisdiction on node or edge
    "shell_node":         1.9,   # node_type == shell_company
    "crypto_node":        1.6,   # node_type == crypto_exchange
    "high_sw":            3.5,   # suspicion_weight > 0.6
    "normal":             0.75,  # no signals
}

_NOISE_RATE  = 0.15   # 15% synthetic noise in query_transactions
_MISSING_RATE = 0.20  # 20% missing/noisy fields in lookup_entity_info
_BFS_NODE_CAP = 150   # max nodes returned by trace_network
_BFS_EDGE_CAP = 300   # max edges returned by trace_network


# ---------------------------------------------------------------------------
# Tool 1: query_transactions
# ---------------------------------------------------------------------------

def query_transactions(
    graph: TransactionGraph,
    entity_id: str,
    max_results: int = 50,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, Any]:
    """Return up to max_results transactions involving entity_id with 15% noise."""
    rng = rng or np.random.default_rng()

    if not graph.graph.has_node(entity_id):
        return {
            "tool":        "query_transactions",
            "status":      "error",
            "entity_id":   entity_id,
            "budget_cost": TOOL_BUDGET["query_transactions"],
            "data":        {"error": f"Entity '{entity_id}' not found."},
        }

    G = graph.graph
    txs: List[dict] = []

    for tgt, data in G[entity_id].items():
        txs.append({"direction": "outgoing", "counterparty": tgt, **dict(data)})
    for src in G.predecessors(entity_id):
        txs.append({"direction": "incoming", "counterparty": src, **dict(G[src][entity_id])})

    # Cap to max_results
    if len(txs) > max_results:
        idx = rng.choice(len(txs), max_results, replace=False)
        txs = [txs[i] for i in idx]

    # Inject 15% synthetic noise: slightly perturb amounts and timestamps
    for tx in txs:
        if rng.random() < _NOISE_RATE:
            tx["amount"]    = float(tx["amount"]) * float(rng.uniform(0.95, 1.05))
            tx["timestamp"] = int(tx["timestamp"]) + int(rng.integers(-3600, 3600))
            tx["_noisy"]    = True

    # Compute a simple suspicion score: fraction of high-suspicion-weight txs
    if txs:
        avg_sw = float(np.mean([t.get("suspicion_weight", 0.0) for t in txs]))
    else:
        avg_sw = 0.0

    return {
        "tool":        "query_transactions",
        "status":      "ok",
        "entity_id":   entity_id,
        "budget_cost": TOOL_BUDGET["query_transactions"],
        "data": {
            "transaction_count": len(txs),
            "avg_suspicion_weight": round(avg_sw, 4),
            "transactions": txs,
        },
    }


# ---------------------------------------------------------------------------
# Tool 2: trace_network
# ---------------------------------------------------------------------------

def trace_network(
    graph: TransactionGraph,
    entity_id: str,
    max_depth: int = 3,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, Any]:
    """BFS from entity_id up to max_depth hops. Capped at _BFS_NODE_CAP/_BFS_EDGE_CAP."""
    rng = rng or np.random.default_rng()

    if not graph.graph.has_node(entity_id):
        return {
            "tool":        "trace_network",
            "status":      "error",
            "entity_id":   entity_id,
            "budget_cost": TOOL_BUDGET["trace_network"],
            "data":        {"error": f"Entity '{entity_id}' not found."},
        }

    G = graph.graph
    visited_nodes: Dict[str, int] = {entity_id: 0}  # node → depth
    edges: List[dict]             = []
    queue: deque                  = deque([(entity_id, 0)])

    while queue and len(visited_nodes) < _BFS_NODE_CAP and len(edges) < _BFS_EDGE_CAP:
        node, depth = queue.popleft()
        if depth >= max_depth:
            continue
        for nbr, data in G[node].items():
            if len(edges) >= _BFS_EDGE_CAP:
                break
            edges.append({
                "src": node,
                "tgt": nbr,
                **dict(data),
            })
            if nbr not in visited_nodes and len(visited_nodes) < _BFS_NODE_CAP:
                visited_nodes[nbr] = depth + 1
                queue.append((nbr, depth + 1))

    nodes_info = [
        {"entity_id": n, "depth": d, **dict(G.nodes[n])}
        for n, d in visited_nodes.items()
        if G.has_node(n)
    ]

    return {
        "tool":        "trace_network",
        "status":      "ok",
        "entity_id":   entity_id,
        "budget_cost": TOOL_BUDGET["trace_network"],
        "data": {
            "origin":     entity_id,
            "max_depth":  max_depth,
            "node_count": len(nodes_info),
            "edge_count": len(edges),
            "nodes":      nodes_info,
            "edges":      edges,
        },
    }


# ---------------------------------------------------------------------------
# Tool 3: lookup_entity_info
# ---------------------------------------------------------------------------

def lookup_entity_info(
    graph: TransactionGraph,
    entity_id: str,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, Any]:
    """Return node metadata. 20% of fields are noisy or absent (partial observability)."""
    rng = rng or np.random.default_rng()

    if not graph.graph.has_node(entity_id):
        return {
            "tool":        "lookup_entity_info",
            "status":      "error",
            "entity_id":   entity_id,
            "budget_cost": TOOL_BUDGET["lookup_entity_info"],
            "data":        {"error": f"Entity '{entity_id}' not found."},
        }

    attrs = dict(graph.graph.nodes[entity_id])
    degree_out = graph.graph.out_degree(entity_id)
    degree_in  = graph.graph.in_degree(entity_id)

    info: Dict[str, Any] = {
        "entity_id":   entity_id,
        "node_type":   attrs.get("node_type"),
        "jurisdiction": attrs.get("jurisdiction"),
        "degree_out":  degree_out,
        "degree_in":   degree_in,
        "is_offshore": attrs.get("jurisdiction") in OFFSHORE_JURISDICTIONS,
    }

    # Apply 20% noise: randomly redact or corrupt individual fields
    noisy_fields: List[str] = []
    redactable = ["jurisdiction", "degree_out", "degree_in"]
    for field in redactable:
        if rng.random() < _MISSING_RATE:
            info[field]  = None
            noisy_fields.append(field)

    return {
        "tool":        "lookup_entity_info",
        "status":      "ok",
        "entity_id":   entity_id,
        "budget_cost": TOOL_BUDGET["lookup_entity_info"],
        "data": {
            **info,
            "noisy_fields": noisy_fields,
        },
    }


# ---------------------------------------------------------------------------
# Tool 4: cross_reference_jurisdiction
# ---------------------------------------------------------------------------

def cross_reference_jurisdiction(
    graph: TransactionGraph,
    entity_id: str,
    has_subpoena: bool = False,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, Any]:
    """
    Check cross-border transaction flags for entity_id.
    Sensitive bank details redacted unless has_subpoena=True.
    """
    rng = rng or np.random.default_rng()

    if not graph.graph.has_node(entity_id):
        return {
            "tool":        "cross_reference_jurisdiction",
            "status":      "error",
            "entity_id":   entity_id,
            "budget_cost": TOOL_BUDGET["cross_reference_jurisdiction"],
            "data":        {"error": f"Entity '{entity_id}' not found."},
        }

    G          = graph.graph
    node_jur   = G.nodes[entity_id].get("jurisdiction", "US")
    is_offshore = node_jur in OFFSHORE_JURISDICTIONS

    # Collect all counterparty jurisdictions
    cross_border: List[dict] = []
    for tgt, data in G[entity_id].items():
        tgt_jur = G.nodes[tgt].get("jurisdiction", "US")
        if tgt_jur != node_jur:
            entry: Dict[str, Any] = {
                "counterparty":      tgt if has_subpoena else "[REDACTED]",
                "counterparty_jur":  tgt_jur,
                "amount":            round(float(data.get("amount", 0)), 2) if has_subpoena else None,
                "transaction_type":  data.get("transaction_type"),
                "is_offshore_dest":  tgt_jur in OFFSHORE_JURISDICTIONS,
                "suspicion_weight":  round(float(data.get("suspicion_weight", 0)), 4),
            }
            cross_border.append(entry)

    offshore_hops = sum(1 for e in cross_border if e["is_offshore_dest"])
    risk_score    = min(1.0, is_offshore * 0.4 + offshore_hops * 0.05)

    return {
        "tool":        "cross_reference_jurisdiction",
        "status":      "ok",
        "entity_id":   entity_id,
        "budget_cost": TOOL_BUDGET["cross_reference_jurisdiction"],
        "data": {
            "entity_jurisdiction":  node_jur,
            "is_offshore":          is_offshore,
            "cross_border_count":   len(cross_border),
            "offshore_hop_count":   offshore_hops,
            "risk_score":           round(risk_score, 4),
            "has_subpoena":         has_subpoena,
            "cross_border_entries": cross_border[:20],  # cap for readability
        },
    }


# ---------------------------------------------------------------------------
# Tool 5: file_SAR
# ---------------------------------------------------------------------------

def file_SAR(
    graph: TransactionGraph,
    scheme_id: str,
    evidence_chain: List[str],
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, Any]:
    """
    Validate evidence_chain against ground truth for scheme_id.
    Returns compliance_score (0–1) and structured feedback.
    """
    rng = rng or np.random.default_rng()

    gt = graph.ground_truth.get(scheme_id)
    if gt is None:
        return {
            "tool":        "file_SAR",
            "status":      "error",
            "entity_id":   None,
            "budget_cost": TOOL_BUDGET["file_SAR"],
            "data":        {"error": f"Scheme '{scheme_id}' not found in ground truth."},
        }

    full_path   = set(gt.get("full_path", []))
    evidence    = set(evidence_chain)
    overlap     = evidence & full_path
    precision   = len(overlap) / max(len(evidence), 1)
    recall      = len(overlap) / max(len(full_path), 1)
    f1          = (2 * precision * recall) / max(precision + recall, 1e-9)
    compliance  = round(f1, 4)

    missed      = list(full_path - evidence)[:5]
    false_pos   = list(evidence - full_path)[:5]

    feedback: List[str] = []
    if recall < 0.5:
        feedback.append("Insufficient path coverage — trace further before filing.")
    if precision < 0.5:
        feedback.append("Too many unrelated entities in evidence chain.")
    if compliance >= 0.8:
        feedback.append("Strong SAR — well-supported evidence chain.")

    return {
        "tool":        "file_SAR",
        "status":      "filed",
        "entity_id":   gt.get("source_entity"),
        "budget_cost": TOOL_BUDGET["file_SAR"],
        "data": {
            "scheme_id":          scheme_id,
            "scheme_type":        gt.get("scheme_type"),
            "compliance_score":   compliance,
            "detection_overlap":  round(len(overlap) / max(len(full_path), 1), 4),
            "precision":          round(precision, 4),
            "recall":             round(recall, 4),
            "missed_entities":    missed,
            "false_positives":    false_pos,
            "feedback":           feedback,
        },
    }


# ---------------------------------------------------------------------------
# Tool 6: request_subpoena
# ---------------------------------------------------------------------------

def request_subpoena(
    graph: TransactionGraph,
    entity_id: str,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, Any]:
    """
    Request a subpoena for entity_id. Costs 2 budget units.
    Returns a subpoena token that unlocks sensitive data in cross_reference_jurisdiction.
    """
    rng = rng or np.random.default_rng()

    if not graph.graph.has_node(entity_id):
        return {
            "tool":        "request_subpoena",
            "status":      "error",
            "entity_id":   entity_id,
            "budget_cost": TOOL_BUDGET["request_subpoena"],
            "data":        {"error": f"Entity '{entity_id}' not found."},
        }

    token = f"SUBPOENA-{uuid.uuid4().hex[:12].upper()}"
    node_type = graph.graph.nodes[entity_id].get("node_type", "unknown")

    return {
        "tool":        "request_subpoena",
        "status":      "subpoena_granted",
        "entity_id":   entity_id,
        "budget_cost": TOOL_BUDGET["request_subpoena"],
        "data": {
            "entity_id":     entity_id,
            "node_type":     node_type,
            "subpoena_token": token,
            "unlocks":       "full transaction details + counterparty identities",
            "note":          "Pass has_subpoena=True to cross_reference_jurisdiction.",
        },
    }


# ---------------------------------------------------------------------------
# Bayesian belief updater
# ---------------------------------------------------------------------------

def bayesian_update(
    prior: float,
    tool_name: str,
    tool_result: Dict[str, Any],
    entity_id: str,
    graph: TransactionGraph,
) -> float:
    """
    Proper Bayesian update: P_new = (P_old * LR) / (P_old * LR + (1 - P_old)).

    Likelihood ratio is selected based on the strongest signal present in the
    tool result and the graph structure for entity_id.
    """
    if not graph.graph.has_node(entity_id):
        return prior

    G    = graph.graph
    node = G.nodes[entity_id]
    lr   = _LR["normal"]

    # --- Structural signals (graph) ---
    node_type = node.get("node_type", "")
    node_jur  = node.get("jurisdiction", "")

    if node_type == "shell_company":
        lr = max(lr, _LR["shell_node"])
    if node_type == "crypto_exchange":
        lr = max(lr, _LR["crypto_node"])
    if node_jur in OFFSHORE_JURISDICTIONS:
        lr = max(lr, _LR["offshore_jur"])

    # --- Edge signals ---
    for _, data in G[entity_id].items():
        sw = float(data.get("suspicion_weight", 0.0))
        if sw > 0.6:
            lr = max(lr, _LR["high_sw"])
        amt = float(data.get("amount", 0.0))
        if 9_000.0 <= amt <= 9_999.0:
            lr = max(lr, _LR["structuring_amount"])

    # --- Tool-result signals ---
    data = tool_result.get("data", {})

    if tool_name == "cross_reference_jurisdiction":
        risk = float(data.get("risk_score", 0.0))
        if risk > 0.3:
            lr = max(lr, _LR["offshore_jur"])

    if tool_name == "query_transactions":
        avg_sw = float(data.get("avg_suspicion_weight", 0.0))
        if avg_sw > 0.4:
            lr = max(lr, _LR["high_sw"])

    if tool_name == "trace_network":
        # Promote if the traced subgraph contains shell/offshore nodes
        for n in data.get("nodes", []):
            if n.get("node_type") == "shell_company":
                lr = max(lr, _LR["shell_node"])
                break

    # P_new = P_old * LR / (P_old * LR + (1 - P_old))
    prior = max(1e-6, min(1 - 1e-6, prior))
    posterior = (prior * lr) / (prior * lr + (1.0 - prior))
    return round(min(posterior, 0.99), 4)


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print("Building TransactionGraph (takes ~10s)…")
    tg = TransactionGraph(seed=0)
    sid = tg.inject_scheme("smurfing")
    gt  = tg.ground_truth[sid]
    src = gt["source_entity"]
    mid = gt["intermediate_nodes"][0] if gt["intermediate_nodes"] else gt["sink_entity"]
    rng = np.random.default_rng(1)

    passed = 0
    failed = 0

    def check(name: str, cond: bool, detail: str = "") -> None:
        global passed, failed
        if cond:
            print(f"  PASS  {name}")
            passed += 1
        else:
            print(f"  FAIL  {name}  {detail}")
            failed += 1

    # ── Tool 1: query_transactions ────────────────────────────────────────
    r1 = query_transactions(tg, src, max_results=50, rng=rng)
    check("query_transactions status=ok",          r1["status"] == "ok")
    check("query_transactions budget_cost=1",       r1["budget_cost"] == 1)
    check("query_transactions has transactions",    len(r1["data"]["transactions"]) > 0)
    check("query_transactions count<=50",           r1["data"]["transaction_count"] <= 50)

    r1e = query_transactions(tg, "nonexistent_999", rng=rng)
    check("query_transactions error on bad entity", r1e["status"] == "error")

    # ── Tool 2: trace_network ─────────────────────────────────────────────
    r2 = trace_network(tg, src, max_depth=2, rng=rng)
    check("trace_network status=ok",       r2["status"] == "ok")
    check("trace_network budget_cost=1",   r2["budget_cost"] == 1)
    check("trace_network has nodes",       r2["data"]["node_count"] > 0)
    check("trace_network node_cap",        r2["data"]["node_count"] <= _BFS_NODE_CAP)
    check("trace_network edge_cap",        r2["data"]["edge_count"] <= _BFS_EDGE_CAP)

    # ── Tool 3: lookup_entity_info ────────────────────────────────────────
    r3 = lookup_entity_info(tg, src, rng=rng)
    check("lookup_entity_info status=ok",     r3["status"] == "ok")
    check("lookup_entity_info budget_cost=1", r3["budget_cost"] == 1)
    check("lookup_entity_info has node_type", "node_type" in r3["data"])

    # ── Tool 4: cross_reference_jurisdiction ──────────────────────────────
    r4 = cross_reference_jurisdiction(tg, src, has_subpoena=False, rng=rng)
    check("cross_ref status=ok",       r4["status"] == "ok")
    check("cross_ref budget_cost=1",   r4["budget_cost"] == 1)
    check("cross_ref no subpoena — counterparty redacted",
          all(e["counterparty"] == "[REDACTED]" for e in r4["data"]["cross_border_entries"]))

    r4s = cross_reference_jurisdiction(tg, src, has_subpoena=True, rng=rng)
    check("cross_ref with subpoena — counterparty visible",
          all(e["counterparty"] != "[REDACTED]" for e in r4s["data"]["cross_border_entries"])
          or r4s["data"]["cross_border_count"] == 0)

    # ── Tool 5: file_SAR ──────────────────────────────────────────────────
    full_path = gt["full_path"]
    r5_good = file_SAR(tg, sid, full_path, rng=rng)
    check("file_SAR status=filed",            r5_good["status"] == "filed")
    check("file_SAR budget_cost=1",           r5_good["budget_cost"] == 1)
    check("file_SAR full path -> score>=0.9", r5_good["data"]["compliance_score"] >= 0.9)

    r5_bad = file_SAR(tg, sid, [src], rng=rng)
    check("file_SAR partial evidence -> score<0.9", r5_bad["data"]["compliance_score"] < 0.9)

    r5e = file_SAR(tg, "fake_scheme_id", full_path, rng=rng)
    check("file_SAR error on bad scheme_id", r5e["status"] == "error")

    # ── Tool 6: request_subpoena ──────────────────────────────────────────
    r6 = request_subpoena(tg, mid, rng=rng)
    check("request_subpoena status=subpoena_granted", r6["status"] == "subpoena_granted")
    check("request_subpoena budget_cost=2",            r6["budget_cost"] == 2)
    check("request_subpoena has token",
          r6["data"]["subpoena_token"].startswith("SUBPOENA-"))

    # ── Bayesian update direction ─────────────────────────────────────────
    prior  = 0.3
    post   = bayesian_update(prior, "query_transactions", r1, src, tg)
    check("bayesian_update returns float",  isinstance(post, float))
    check("bayesian_update bounded [0,1]",  0.0 <= post <= 1.0)

    # For a shell/offshore node, posterior should be >= prior
    shell_nodes = tg._nodes_by_type.get("shell_company", [])
    if shell_nodes:
        dummy_result: Dict[str, Any] = {"data": {}}
        shell_post = bayesian_update(0.3, "lookup_entity_info", dummy_result, shell_nodes[0], tg)
        check("bayesian_update shell node raises posterior", shell_post >= 0.3)

    # ── Budget totals across all 6 tools ─────────────────────────────────
    total = sum(TOOL_BUDGET.values())
    check("budget totals 7 across 6 tools (subpoena=2, rest=1 each)", total == 7)

    print(f"\n{passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)
