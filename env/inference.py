"""
inference.py — HEIST heuristic inference client.

Runs a full episode against the HEIST OpenEnv server:
    reset → observe → choose action → POST /step → repeat until done=True

Action selection (deterministic heuristic):
    AlertTriage   → query_transactions on highest-belief entity
    Investigation → trace_network on highest-belief entity
    CrossReference → cross_reference_jurisdiction (subpoena first if budget allows)
    SARFiling     → file_SAR with accumulated evidence chain

Supports environment variables:
    ENV_BASE_URL   (default: http://localhost:8000)
    API_BASE_URL   (optional: LLM endpoint for non-heuristic mode)
    MODEL_NAME     (optional: LLM model name)
    HF_TOKEN       (optional)
    OPENAI_API_KEY (optional)

Usage:
    # Start server first:  uv run --project . server
    # Then run inference:
    python inference.py
    python inference.py --seed 42
    python inference.py --max-steps 30
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "http://localhost:8000")
API_BASE_URL = os.environ.get("API_BASE_URL", "")  # optional LLM
MODEL_NAME   = os.environ.get("MODEL_NAME", "")
HF_TOKEN     = os.environ.get("HF_TOKEN", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _post(path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """POST JSON to the HEIST server and return parsed response."""
    url = f"{ENV_BASE_URL}{path}"
    resp = requests.post(url, json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()


def reset_episode(seed: int = 42) -> Dict[str, Any]:
    """POST /reset → initial observation."""
    return _post("/reset", {"seed": seed})


def step_action(action_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """POST /step → observation after executing one tool."""
    return _post("/step", {
        "action_type": action_type,
        "params": params,
    })


# ---------------------------------------------------------------------------
# Heuristic helpers
# ---------------------------------------------------------------------------

def _top_belief_entity(beliefs: Dict[str, float]) -> Optional[str]:
    """Return entity_id with highest P(criminal) belief."""
    if not beliefs:
        return None
    return max(beliefs, key=beliefs.get)


def _entities_from_visible(visible: List[Dict[str, Any]]) -> List[str]:
    """Extract entity_ids from visible_entities list."""
    return [e["entity_id"] for e in visible if "entity_id" in e]


def _extract_entities_from_obs(obs: Dict[str, Any]) -> List[str]:
    """Extract all entity_ids from observation (visible + tool_result + flagged)."""
    entities: List[str] = []

    # From visible entities
    for e in obs.get("visible_entities", []):
        eid = e.get("entity_id")
        if eid:
            entities.append(eid)

    # From flagged transaction
    flagged = obs.get("flagged_transaction") or {}
    for key in ("source_entity", "target_entity"):
        eid = flagged.get(key)
        if eid:
            entities.append(eid)

    # From tool_result data (trace_network, query_transactions, etc.)
    tool_result = obs.get("tool_result") or {}
    data = tool_result.get("data", {})

    # trace_network returns nodes with entity_id
    for node in data.get("nodes", []):
        eid = node.get("entity_id")
        if eid:
            entities.append(eid)

    # query_transactions returns transactions with counterparty
    for tx in data.get("transactions", []):
        cp = tx.get("counterparty")
        if cp:
            entities.append(cp)

    # cross_reference returns cross_border_entries with counterparty
    for entry in data.get("cross_border_entries", []):
        cp = entry.get("counterparty")
        if cp and cp != "[REDACTED]":
            entities.append(cp)

    return list(dict.fromkeys(entities))  # dedupe preserving order


def select_action(
    obs: Dict[str, Any],
    evidence_chain: List[str],
    subpoenaed: set,
    step_num: int,
) -> tuple[str, Dict[str, Any]]:
    """
    Deterministic heuristic action selection based on current phase.

    Returns (action_type, params) tuple.
    """
    phase   = obs.get("current_phase", "AlertTriage")
    beliefs = obs.get("bayesian_beliefs", {})
    visible = obs.get("visible_entities", [])
    budget  = obs.get("budget_remaining", 0)
    flagged = obs.get("flagged_transaction")

    # Pick the entity we're most suspicious of
    target = _top_belief_entity(beliefs)

    # Fallback: use flagged transaction source or first visible entity
    if not target:
        if flagged:
            target = flagged.get("source_entity", "")
        elif visible:
            target = visible[0].get("entity_id", "")

    if not target:
        # Absolute fallback — file SAR with whatever we have
        return "file_SAR", {"evidence_chain": evidence_chain}

    # ── Phase-based heuristic ──────────────────────────────────────────

    if phase == "AlertTriage":
        return "query_transactions", {"entity_id": target}

    if phase == "Investigation":
        return "trace_network", {"entity_id": target, "max_depth": 3}

    if phase == "CrossReference":
        # Subpoena the target first if we haven't and have budget
        if target not in subpoenaed and budget >= 3:
            return "request_subpoena", {"entity_id": target}
        return "cross_reference_jurisdiction", {"entity_id": target}

    if phase == "SARFiling":
        # Collect all high-belief entities into evidence chain
        high_belief = [
            eid for eid, p in beliefs.items() if p > 0.3
        ]
        chain = list(set(evidence_chain + high_belief))
        return "file_SAR", {"evidence_chain": chain}

    # Unknown phase — safe fallback
    return "query_transactions", {"entity_id": target}


# ---------------------------------------------------------------------------
# Optional LLM action selection (stub — used when API_BASE_URL is set)
# ---------------------------------------------------------------------------

def _try_llm_action(obs: Dict[str, Any]) -> Optional[tuple[str, Dict[str, Any]]]:
    """
    Call an external LLM to choose the next action.
    Returns None if no LLM is configured or the call fails.
    Falls back to heuristic if anything goes wrong.
    """
    if not API_BASE_URL or not MODEL_NAME:
        return None

    try:
        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if OPENAI_API_KEY:
            headers["Authorization"] = f"Bearer {OPENAI_API_KEY}"
        elif HF_TOKEN:
            headers["Authorization"] = f"Bearer {HF_TOKEN}"

        prompt = (
            f"You are an AML investigator. Current phase: {obs.get('current_phase')}. "
            f"Budget: {obs.get('budget_remaining')}. "
            f"Beliefs: {json.dumps(obs.get('bayesian_beliefs', {}))}. "
            f"Choose ONE action: query_transactions, trace_network, lookup_entity, "
            f"cross_reference_jurisdiction, file_SAR, or request_subpoena. "
            f"Reply with JSON: {{\"action_type\": \"...\", \"params\": {{...}}}}"
        )

        resp = requests.post(
            f"{API_BASE_URL}/v1/chat/completions",
            headers=headers,
            json={
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 200,
                "temperature": 0.1,
            },
            timeout=30,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        parsed = json.loads(content)
        return parsed["action_type"], parsed.get("params", {})
    except Exception:
        return None  # fallback to heuristic


# ---------------------------------------------------------------------------
# Main episode loop
# ---------------------------------------------------------------------------

def run_episode(seed: int = 42, max_steps: int = 50, verbose: bool = True) -> Dict[str, Any]:
    """
    Run one complete HEIST episode.

    Returns dict with final metrics:
        compliance_score, termination_reason, steps_used, budget_used, evidence_chain
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"HEIST Inference — seed={seed}, max_steps={max_steps}")
        print(f"{'='*60}")

    # ── Reset ──────────────────────────────────────────────────────────
    obs = reset_episode(seed)
    if verbose:
        print(f"\n[RESET] Phase={obs.get('current_phase')}  "
              f"Budget={obs.get('budget_remaining')}  "
              f"Scheme={obs.get('metadata', {}).get('scheme_type', '?')}")
        flagged = obs.get("flagged_transaction")
        if flagged:
            print(f"  Flagged: {flagged.get('source_entity')} → {flagged.get('target_entity')}")

    evidence_chain: List[str] = []
    subpoenaed: set[str] = set()
    step = 0
    final_compliance = 0.0
    final_reason = None

    # ── Step loop ──────────────────────────────────────────────────────
    while not obs.get("done", False) and step < max_steps:
        step += 1

        # Accumulate entities we've seen into evidence chain
        for eid in _extract_entities_from_obs(obs):
            if eid not in evidence_chain:
                evidence_chain.append(eid)

        # Try LLM first, fall back to heuristic
        llm_action = _try_llm_action(obs)
        if llm_action:
            action_type, params = llm_action
        else:
            action_type, params = select_action(obs, evidence_chain, subpoenaed, step)

        # Track subpoenas
        if action_type == "request_subpoena":
            eid = params.get("entity_id", "")
            if eid:
                subpoenaed.add(eid)

        # Execute
        obs = step_action(action_type, params)

        # Extract info for logging
        phase   = obs.get("current_phase", "?")
        reward  = obs.get("reward", 0.0)
        budget  = obs.get("budget_remaining", 0)
        beliefs = obs.get("bayesian_beliefs", {})
        top_eid = _top_belief_entity(beliefs)
        top_p   = beliefs.get(top_eid, 0.0) if top_eid else 0.0

        if verbose:
            print(f"  [{step:2d}] {phase:15s} | {action_type:30s} | "
                  f"R={reward:+6.2f} | Budget={budget:3d} | "
                  f"Top={top_eid}  P={top_p:.4f}")

        # Check morph
        morph = obs.get("morph_alert", {})
        if morph.get("morph_occurred"):
            if verbose:
                inv = morph.get("invalidated_entities", [])
                print(f"  *** MORPH DETECTED — invalidated: {inv}")

        # Track termination
        reason = obs.get("termination_reason")
        if reason:
            final_reason = reason

        # Extract compliance score from SAR filing
        tool_result = obs.get("tool_result") or {}
        data = tool_result.get("data", {})
        if "compliance_score" in data:
            final_compliance = data["compliance_score"]

    # ── Summary ────────────────────────────────────────────────────────
    budget_used = 50 - obs.get("budget_remaining", 0)
    result = {
        "compliance_score":   final_compliance,
        "termination_reason": final_reason,
        "steps_used":         step,
        "budget_used":        budget_used,
        "evidence_chain":     evidence_chain,
        "final_beliefs":      obs.get("bayesian_beliefs", {}),
    }

    if verbose:
        print(f"\n{'='*60}")
        print(f"EPISODE COMPLETE")
        print(f"  Termination:      {final_reason}")
        print(f"  Compliance Score: {final_compliance:.4f}")
        print(f"  Steps Used:       {step}")
        print(f"  Budget Used:      {budget_used}")
        print(f"  Evidence Chain:   {len(evidence_chain)} entities")
        print(f"{'='*60}\n")

    return result


# ---------------------------------------------------------------------------
# Direct-call mode (no server, uses HeistEnvironment directly)
# ---------------------------------------------------------------------------

def run_episode_direct(seed: int = 42, max_steps: int = 50, verbose: bool = True) -> Dict[str, Any]:
    """
    Run one complete HEIST episode WITHOUT requiring the server.
    Calls HeistEnvironment directly — useful for smoke testing and training.
    """
    # Import locally to avoid circular issues when used as a module
    try:
        from heist_env import HeistEnvironment
        from models import InvestigatorAction, ActionType
    except ImportError:
        from env.heist_env import HeistEnvironment
        from env.models import InvestigatorAction, ActionType

    env = HeistEnvironment()

    if verbose:
        print(f"\n{'='*60}")
        print(f"HEIST Direct Inference — seed={seed}")
        print(f"{'='*60}")

    obs = env.reset(seed=seed)
    obs_dict = obs.model_dump() if hasattr(obs, "model_dump") else obs.dict()

    if verbose:
        print(f"\n[RESET] Phase={obs_dict.get('current_phase')}  "
              f"Budget={obs_dict.get('budget_remaining')}  "
              f"Scheme={obs_dict.get('metadata', {}).get('scheme_type', '?')}")

    evidence_chain: List[str] = []
    subpoenaed: set[str] = set()
    step = 0
    final_compliance = 0.0
    final_reason = None

    while not obs_dict.get("done", False) and step < max_steps:
        step += 1

        # Accumulate entities
        for eid in _extract_entities_from_obs(obs_dict):
            if eid not in evidence_chain:
                evidence_chain.append(eid)

        # Heuristic action selection
        action_type_str, params = select_action(obs_dict, evidence_chain, subpoenaed, step)

        # Track subpoenas
        if action_type_str == "request_subpoena":
            eid = params.get("entity_id", "")
            if eid:
                subpoenaed.add(eid)

        # Map string to ActionType enum
        _type_map = {
            "query_transactions":         ActionType.QUERY_TRANSACTIONS,
            "trace_network":              ActionType.TRACE_NETWORK,
            "lookup_entity":              ActionType.LOOKUP_ENTITY,
            "cross_reference_jurisdiction": ActionType.CROSS_REFERENCE,
            "file_SAR":                   ActionType.FILE_SAR,
            "request_subpoena":           ActionType.REQUEST_SUBPOENA,
        }
        action = InvestigatorAction(
            action_type=_type_map[action_type_str],
            params=params,
        )

        obs = env.step(action)
        obs_dict = obs.model_dump() if hasattr(obs, "model_dump") else obs.dict()

        phase  = obs_dict.get("current_phase", "?")
        reward = obs_dict.get("reward", 0.0)
        budget = obs_dict.get("budget_remaining", 0)
        beliefs = obs_dict.get("bayesian_beliefs", {})
        top_eid = _top_belief_entity(beliefs)
        top_p   = beliefs.get(top_eid, 0.0) if top_eid else 0.0

        if verbose:
            print(f"  [{step:2d}] {phase:15s} | {action_type_str:30s} | "
                  f"R={reward:+6.2f} | Budget={budget:3d} | "
                  f"Top={top_eid}  P={top_p:.4f}")

        morph = obs_dict.get("morph_alert", {})
        if morph.get("morph_occurred"):
            if verbose:
                print(f"  *** MORPH DETECTED — invalidated: {morph.get('invalidated_entities', [])}")

        reason = obs_dict.get("termination_reason")
        if reason:
            final_reason = reason

        tool_result = obs_dict.get("tool_result") or {}
        data = tool_result.get("data", {})
        if "compliance_score" in data:
            final_compliance = data["compliance_score"]

    budget_used = 50 - obs_dict.get("budget_remaining", 0)
    result = {
        "compliance_score":   final_compliance,
        "termination_reason": final_reason,
        "steps_used":         step,
        "budget_used":        budget_used,
        "evidence_chain":     evidence_chain,
        "final_beliefs":      obs_dict.get("bayesian_beliefs", {}),
    }

    if verbose:
        print(f"\n{'='*60}")
        print(f"EPISODE COMPLETE")
        print(f"  Termination:      {final_reason}")
        print(f"  Compliance Score: {final_compliance:.4f}")
        print(f"  Steps Used:       {step}")
        print(f"  Budget Used:      {budget_used}")
        print(f"  Evidence Chain:   {len(evidence_chain)} entities")
        print(f"{'='*60}\n")

    return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="HEIST inference — run one episode")
    parser.add_argument("--seed",      type=int, default=42,    help="Episode seed")
    parser.add_argument("--max-steps", type=int, default=50,    help="Max steps per episode")
    parser.add_argument("--direct",    action="store_true",     help="Skip server, call env directly")
    parser.add_argument("--quiet",     action="store_true",     help="Suppress step-by-step output")
    args = parser.parse_args()

    verbose = not args.quiet

    if args.direct:
        result = run_episode_direct(seed=args.seed, max_steps=args.max_steps, verbose=verbose)
    else:
        result = run_episode(seed=args.seed, max_steps=args.max_steps, verbose=verbose)

    # Exit code: 0 if compliance > 0, 1 otherwise
    sys.exit(0 if result["compliance_score"] > 0.0 else 1)


if __name__ == "__main__":
    main()
