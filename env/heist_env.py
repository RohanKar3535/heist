"""
HeistEnvironment — OpenEnv core for HEIST.

Phase order:  AlertTriage -> Investigation -> CrossReference -> SARFiling
Termination:  file_SAR() called  |  budget exhausted  |  50 steps reached
Morph trigger: max(P(criminal)) > 0.7 and morph_count == 0 (Stackelberg move)
Partial obs:  investigator sees flagged transaction + direct neighbors only.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple
from uuid import uuid4

import numpy as np
from openenv.core.env_server.interfaces import Environment

try:
    from .models import (
        ActionType,
        HeistObservation,
        HeistState,
        InvestigatorAction,
        MorphAlert,
    )
    from .transaction_graph import TransactionGraph, SCHEME_TYPES
    from .tools import (
        query_transactions,
        trace_network,
        lookup_entity_info,
        cross_reference_jurisdiction,
        file_SAR,
        request_subpoena,
        bayesian_update,
    )
except ImportError:
    from models import (
        ActionType,
        HeistObservation,
        HeistState,
        InvestigatorAction,
        MorphAlert,
    )
    from transaction_graph import TransactionGraph, SCHEME_TYPES
    from tools import (
        query_transactions,
        trace_network,
        lookup_entity_info,
        cross_reference_jurisdiction,
        file_SAR,
        request_subpoena,
        bayesian_update,
    )

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_STEPS   = 50
_MAX_BUDGET  = 50
_PHASES      = ["AlertTriage", "Investigation", "CrossReference", "SARFiling"]

# Minimum uses of the trigger action before phase advances
_PHASE_MIN_CALLS: dict[str, int] = {
    "AlertTriage":   2,
    "Investigation": 5,
    "CrossReference": 3,
}

# Which action type triggers advancement out of each phase
_PHASE_TRIGGER: dict[str, ActionType] = {
    "AlertTriage":   ActionType.QUERY_TRANSACTIONS,
    "Investigation": ActionType.TRACE_NETWORK,
    "CrossReference": ActionType.CROSS_REFERENCE,
}

_PHASE_NEXT: dict[str, str] = {
    "AlertTriage":   "Investigation",
    "Investigation": "CrossReference",
    "CrossReference": "SARFiling",
}


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class HeistEnvironment(Environment):
    """
    HEIST RL environment.

    One episode = one injected laundering scheme hidden in 100k-node graph.
    The investigator uses 6 tools to trace the scheme and file a SAR.
    The criminal designer may morph the scheme once if exposed (P > 0.7).
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self) -> None:
        super().__init__()
        self._graph = TransactionGraph(seed=42)
        self._state = HeistState()
        self._rng = np.random.default_rng(42)
        self._subpoenaed: set[str] = set()  # entities with active subpoenas
        self._stale_entities: set[str] = set()  # entities with stale trace results
        self._episode_log: list[dict] = []  # episode event log for oversight agent
        self._morph_success: bool = False  # True if morph happened before SAR

    # ------------------------------------------------------------------
    # OpenEnv interface
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> HeistObservation:
        self._reset_rubric()
        self._graph.reset()
        self._subpoenaed = set()
        self._stale_entities = set()
        self._episode_log = []
        self._morph_success = False

        rng = np.random.default_rng(seed if seed is not None else 42)
        self._rng = rng
        scheme_type = str(rng.choice(SCHEME_TYPES))
        scheme_id   = self._graph.inject_scheme(scheme_type)
        gt          = self._graph.ground_truth[scheme_id]
        flagged_entity = gt["source_entity"]

        self._state = HeistState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            scheme_id=scheme_id,
            scheme_type=scheme_type,
            active_scheme_injected=True,
            current_phase="AlertTriage",
            query_history=[],
            budget_remaining=_MAX_BUDGET,
            bayesian_beliefs={flagged_entity: 0.3},
            morph_occurred=False,
            morph_count=0,
        )

        self._episode_log.append({
            "event": "episode_start",
            "step": 0,
            "scheme_id": scheme_id,
            "scheme_type": scheme_type,
        })

        return HeistObservation(
            done=False,
            reward=0.0,
            current_phase="AlertTriage",
            step_number=0,
            budget_remaining=_MAX_BUDGET,
            flagged_transaction=self._get_flagged_transaction(),
            visible_entities=self._get_visible_entities(flagged_entity),
            tool_result=None,
            bayesian_beliefs=dict(self._state.bayesian_beliefs),
            morph_alert=MorphAlert(),
            termination_reason=None,
            metadata={
                "episode_id": self._state.episode_id,
                "scheme_type": scheme_type,
            },
        )

    def step(
        self,
        action: InvestigatorAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> HeistObservation:
        st = self._state

        # Guard: episode already over
        if st.budget_remaining <= 0 or st.step_count >= _MAX_STEPS:
            return self._terminal_obs("already_terminated")

        st.step_count      += 1
        st.budget_remaining -= 1

        # Record action
        st.query_history.append({
            "step":        st.step_count,
            "action_type": action.action_type.value,
            "params":      dict(action.params),
        })

        # Dispatch tool
        tool_result, reward = self._dispatch(action)

        # Mark stale trace results for morphed entities
        if self._stale_entities and tool_result:
            tool_result = self._apply_staleness(action, tool_result)

        # Update Bayesian beliefs
        self._bayesian_update(action, tool_result)

        # Phase transitions
        self._maybe_advance_phase(action)

        # Morph trigger: P(criminal) > 0.7, at most 1 morph per episode
        morph_alert = MorphAlert()
        if (
            st.active_scheme_injected
            and not st.morph_occurred
            and st.bayesian_beliefs
            and max(st.bayesian_beliefs.values()) > 0.7
        ):
            morph_alert = self._trigger_morph()

        # Termination check
        done:   bool          = False
        reason: Optional[str] = None

        if action.action_type == ActionType.FILE_SAR:
            done, reason = True, "sar_filed"
        elif st.budget_remaining <= 0:
            done, reason = True, "budget_exhausted"
        elif st.step_count >= _MAX_STEPS:
            done, reason = True, "max_steps_reached"

        return HeistObservation(
            done=done,
            reward=float(reward),
            current_phase=st.current_phase,
            step_number=st.step_count,
            budget_remaining=st.budget_remaining,
            flagged_transaction=self._get_flagged_transaction(),
            visible_entities=self._get_visible_from_history(),
            tool_result=tool_result,
            bayesian_beliefs=dict(st.bayesian_beliefs),
            morph_alert=morph_alert,
            termination_reason=reason,
            metadata={
                "episode_id": st.episode_id,
                "morph_success": self._morph_success,
                "episode_log": self._episode_log,
            },
        )

    @property
    def state(self) -> HeistState:
        return self._state

    def close(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Tool dispatcher — wired to tools.py implementations
    # ------------------------------------------------------------------

    def _dispatch(
        self, action: InvestigatorAction
    ) -> Tuple[dict, float]:
        atype     = action.action_type
        entity_id = action.params.get("entity_id", "")

        if atype == ActionType.QUERY_TRANSACTIONS:
            result = query_transactions(
                self._graph, entity_id, rng=self._rng,
            )
            return result, 0.1

        if atype == ActionType.TRACE_NETWORK:
            max_depth = int(action.params.get("max_depth", 3))
            result = trace_network(
                self._graph, entity_id, max_depth=max_depth, rng=self._rng,
            )
            return result, 0.1

        if atype == ActionType.LOOKUP_ENTITY:
            result = lookup_entity_info(
                self._graph, entity_id, rng=self._rng,
            )
            return result, 0.1

        if atype == ActionType.CROSS_REFERENCE:
            has_subpoena = entity_id in self._subpoenaed
            result = cross_reference_jurisdiction(
                self._graph, entity_id,
                has_subpoena=has_subpoena, rng=self._rng,
            )
            return result, 0.1

        if atype == ActionType.FILE_SAR:
            evidence_chain = action.params.get("evidence_chain", [])
            result = file_SAR(
                self._graph,
                self._state.scheme_id,
                evidence_chain,
                rng=self._rng,
            )
            reward = 10.0 * result["data"].get("compliance_score", 0.0)
            return result, reward

        if atype == ActionType.REQUEST_SUBPOENA:
            result = request_subpoena(
                self._graph, entity_id, rng=self._rng,
            )
            # Subpoena costs 2 total: step() already decremented 1,
            # so we decrement 1 more here (guarded against going below 0).
            if self._state.budget_remaining > 0:
                self._state.budget_remaining -= 1
            # Track the subpoena grant for cross_reference_jurisdiction
            if result.get("status") == "subpoena_granted":
                self._subpoenaed.add(entity_id)
            return result, 0.5

        # Fallback: unknown action type
        return {
            "action": atype.value,
            "params": action.params,
            "note":   "Unknown action type.",
        }, 0.0

    # ------------------------------------------------------------------
    # Bayesian belief update — delegates to tools.bayesian_update()
    # ------------------------------------------------------------------

    def _bayesian_update(
        self, action: InvestigatorAction, tool_result: dict
    ) -> None:
        """
        Proper Bayesian update via tools.bayesian_update().
        Uses mutual-information-maximising likelihood ratios from graph signals.
        """
        eid = action.params.get("entity_id")
        if not eid or not self._graph.graph.has_node(eid):
            return

        prior = self._state.bayesian_beliefs.get(eid, 0.1)
        tool_name = action.action_type.value
        posterior = bayesian_update(prior, tool_name, tool_result, eid, self._graph)
        self._state.bayesian_beliefs[eid] = posterior

    # ------------------------------------------------------------------
    # Phase transitions
    # ------------------------------------------------------------------

    def _maybe_advance_phase(self, action: InvestigatorAction) -> None:
        curr = self._state.current_phase
        trigger = _PHASE_TRIGGER.get(curr)
        if trigger is None or action.action_type != trigger:
            return
        min_calls = _PHASE_MIN_CALLS.get(curr, 2)
        calls_so_far = sum(
            1 for h in self._state.query_history
            if h["action_type"] == trigger.value
        )
        if calls_so_far >= min_calls:
            self._state.current_phase = _PHASE_NEXT[curr]

    # ------------------------------------------------------------------
    # Morph (Stackelberg criminal response)
    # ------------------------------------------------------------------

    def _trigger_morph(self) -> MorphAlert:
        """
        Criminal reroutes 2-3 edges through a new intermediary node in
        a new jurisdiction when investigator belief > 0.7.

        - Reroutes edges through new intermediary (not full scheme replacement)
        - 50% of trace_network results on morphed entities marked stale
        - morph_success_bonus if morph triggers before SAR filed
        - Maximum 1 morph per episode (D4 in DECISIONS.md)
        - Morph event logged to episode_log for oversight agent
        """
        st = self._state
        st.morph_occurred = True
        st.morph_count   += 1

        # Identify the key entity (highest belief) that triggered the morph
        key_entity = max(st.bayesian_beliefs, key=st.bayesian_beliefs.get)

        old_gt = self._graph.ground_truth.get(st.scheme_id, {})
        old_path = list(old_gt.get("full_path", []))
        old_intermediates = list(old_gt.get("intermediate_nodes", []))

        # Find 2-3 edges to reroute (edges involving the key entity or nearby)
        rng = np.random.default_rng(st.step_count + 9999)
        injected_info = self._graph._injected.get(st.scheme_id, {"edges": []})
        scheme_edges = injected_info["edges"]

        # Pick 2-3 edges to reroute
        n_reroute = min(int(rng.integers(2, 4)), len(scheme_edges))
        if n_reroute == 0:
            # Fallback: just mark entities stale
            invalidated = old_intermediates[:3]
            self._stale_entities.update(invalidated)
            self._morph_success = True
            self._log_morph_event(key_entity, invalidated)
            return MorphAlert(morph_occurred=True, invalidated_entities=invalidated)

        reroute_indices = rng.choice(len(scheme_edges), min(n_reroute, len(scheme_edges)), replace=False).tolist()

        # Pick a new intermediary from a different jurisdiction
        current_jurisdictions = set()
        for node in old_path:
            if self._graph.graph.has_node(node):
                current_jurisdictions.add(
                    self._graph.graph.nodes[node].get("jurisdiction", "US")
                )

        # Find shell companies in different jurisdictions
        candidates = [
            n for n, d in self._graph.graph.nodes(data=True)
            if d.get("node_type") in ("shell_company", "account")
            and d.get("jurisdiction", "US") not in current_jurisdictions
            and n not in old_path
        ]
        if not candidates:
            candidates = [
                n for n, d in self._graph.graph.nodes(data=True)
                if d.get("node_type") in ("shell_company", "account")
                and n not in old_path
            ]

        invalidated: List[str] = []

        if candidates:
            new_intermediary = str(rng.choice(candidates))
            new_jurisdiction = self._graph.graph.nodes[new_intermediary].get("jurisdiction", "XX")

            # Reroute selected edges through the new intermediary
            for idx in reroute_indices:
                if idx >= len(scheme_edges):
                    continue
                rec = scheme_edges[idx]
                src, tgt = rec["src"], rec["tgt"]

                # Record the entity being rerouted away from as invalidated
                if tgt in old_intermediates:
                    invalidated.append(tgt)
                elif src in old_intermediates:
                    invalidated.append(src)

                # Remove old edge
                if self._graph.graph.has_edge(src, tgt):
                    old_attrs = dict(self._graph.graph[src][tgt])
                    self._graph.graph.remove_edge(src, tgt)
                    # Add edge src -> new_intermediary
                    self._graph.graph.add_edge(src, new_intermediary, **{
                        **old_attrs,
                        "jurisdiction": new_jurisdiction,
                        "timestamp": old_attrs.get("timestamp", 0) + 3600,
                    })
                    # Add edge new_intermediary -> tgt
                    self._graph.graph.add_edge(new_intermediary, tgt, **{
                        **old_attrs,
                        "amount": old_attrs.get("amount", 0) * float(rng.uniform(0.95, 1.0)),
                        "timestamp": old_attrs.get("timestamp", 0) + 7200,
                        "jurisdiction": self._graph.graph.nodes[tgt].get("jurisdiction", "US"),
                    })

            # Update ground truth path to include new intermediary
            new_gt = dict(old_gt)
            new_path = list(old_path)
            # Insert new intermediary after source if possible
            if len(new_path) >= 2:
                new_path.insert(1, new_intermediary)
            new_gt["full_path"] = new_path
            new_gt["intermediate_nodes"] = list(old_intermediates) + [new_intermediary]
            new_gt["num_hops"] = len(new_path) - 1
            self._graph.ground_truth[st.scheme_id] = new_gt

        # Ensure we have invalidated entities (at least from old intermediates)
        if not invalidated:
            invalidated = old_intermediates[:3]

        # Deduplicate
        invalidated = list(dict.fromkeys(invalidated))

        # Mark 50% of trace results on morphed entities as stale
        self._stale_entities.update(invalidated)

        # morph_success_bonus: True since morph triggered before SAR
        self._morph_success = True

        # Log event for oversight agent
        self._log_morph_event(key_entity, invalidated)

        return MorphAlert(morph_occurred=True, invalidated_entities=invalidated)

    def _log_morph_event(self, key_entity: str, invalidated: List[str]) -> None:
        """Log morph event to episode_log for oversight agent."""
        self._episode_log.append({
            "event": "morph_triggered",
            "step": self._state.step_count,
            "key_entity": key_entity,
            "trigger_belief": self._state.bayesian_beliefs.get(key_entity, 0),
            "invalidated_entities": invalidated,
            "morph_count": self._state.morph_count,
            "sar_filed": False,
        })

    def _apply_staleness(
        self, action: InvestigatorAction, tool_result: dict
    ) -> dict:
        """
        Mark 50% of trace_network results on morphed entities as stale.
        Stale entries get is_stale=True flag so investigator knows evidence
        may be unreliable.
        """
        if action.action_type != ActionType.TRACE_NETWORK:
            return tool_result

        entity_id = action.params.get("entity_id", "")
        if entity_id not in self._stale_entities:
            return tool_result

        data = tool_result.get("data", {})
        edges = data.get("edges", [])
        nodes = data.get("nodes", [])

        if not edges:
            return tool_result

        # Mark 50% of edges as stale
        rng = np.random.default_rng(self._state.step_count)
        n_stale = max(1, len(edges) // 2)
        stale_indices = set(rng.choice(len(edges), n_stale, replace=False).tolist())

        for i, edge in enumerate(edges):
            edge["is_stale"] = i in stale_indices

        # Mark corresponding nodes
        stale_node_ids = set()
        for i in stale_indices:
            if i < len(edges):
                stale_node_ids.add(edges[i].get("src", ""))
                stale_node_ids.add(edges[i].get("tgt", ""))

        for node in nodes:
            node["is_stale"] = node.get("entity_id", "") in stale_node_ids

        tool_result["data"]["edges"] = edges
        tool_result["data"]["nodes"] = nodes
        tool_result["data"]["stale_warning"] = (
            "Evidence on this entity may be stale due to criminal morphing. "
            f"{n_stale}/{len(edges)} edges marked as potentially invalidated."
        )

        return tool_result

    # ------------------------------------------------------------------
    # Observation helpers
    # ------------------------------------------------------------------

    def _get_flagged_transaction(self) -> Optional[dict]:
        sid = self._state.scheme_id
        if not sid or sid not in self._graph._injected:
            return None
        edges = self._graph._injected[sid]["edges"]
        if not edges:
            return None
        rec = edges[0]
        src, tgt = rec["src"], rec["tgt"]
        if self._graph.graph.has_edge(src, tgt):
            return {
                "source_entity": src,
                "target_entity": tgt,
                **dict(self._graph.graph[src][tgt]),
                "flagged": True,
            }
        return None

    def _get_visible_entities(self, entity_id: str) -> List[dict]:
        if not self._graph.graph.has_node(entity_id):
            return []
        return [{"entity_id": entity_id, **dict(self._graph.graph.nodes[entity_id])}]

    def _get_visible_from_history(self) -> List[dict]:
        seen: List[dict] = []
        for h in self._state.query_history[-5:]:
            eid = h.get("params", {}).get("entity_id")
            if (
                eid
                and self._graph.graph.has_node(eid)
                and not any(e["entity_id"] == eid for e in seen)
            ):
                seen.append(
                    {"entity_id": eid, **dict(self._graph.graph.nodes[eid])}
                )
        return seen

    def _terminal_obs(self, reason: str) -> HeistObservation:
        return HeistObservation(
            done=True,
            reward=0.0,
            current_phase=self._state.current_phase,
            step_number=self._state.step_count,
            budget_remaining=self._state.budget_remaining,
            flagged_transaction=None,
            visible_entities=[],
            tool_result=None,
            bayesian_beliefs=dict(self._state.bayesian_beliefs),
            morph_alert=MorphAlert(),
            termination_reason=reason,
            metadata={},
        )
