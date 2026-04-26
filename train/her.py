"""
Hindsight Experience Replay (HER) for LLM Tool-Use Agents — Step 17.

Adapted from Andrychowicz et al., 2017 ("Hindsight Experience Replay") for
the HEIST multi-step investigation setting.

Core insight
------------
Most HEIST episodes fail (F1 < 0.5) because the agent investigates the wrong
part of the graph.  Standard GRPO discards these episodes — they contribute
near-zero gradient signal.  HER recycles them:

    1. Take the agent's actual investigation path (evidence_chain).
    2. Find a connected subgraph of visited entities in the transaction graph.
    3. Pretend *that* subgraph was the criminal scheme ("virtual ground truth").
    4. Recompute R_investigator with the relabeled goal → high F1 (~0.8+).
    5. Feed the relabeled rollout into the GRPO group alongside real rollouts.

The agent learns *how to investigate coherently* even from failed episodes,
because the investigation strategy (query → trace → cross-ref → SAR) was
correct — it just targeted the wrong part of the graph.

Why this is novel
-----------------
HER has been applied to robotic manipulation (reach goals) and navigation
(reach locations), but **never to LLM agents with tool-use**.  The key
adaptation is that "goals" in HEIST are *subgraphs* (criminal scheme paths),
not scalar positions — so virtual goal selection must respect graph topology.

Parameters
----------
k_hindsight : int   — number of virtual relabelings per failed episode (default 3)
f1_threshold : float — F1 below which an episode is considered "failed" (default 0.5)
max_group_size : int — cap on total GRPO group size after HER expansion (default 16)
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_ENV  = os.path.join(_ROOT, "env")
for _p in [_ROOT, _ENV]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from reward import r_investigator


# ---------------------------------------------------------------------------
# HER Buffer
# ---------------------------------------------------------------------------

class HindsightReplayBuffer:
    """
    Generates hindsight rollouts from failed HEIST episodes.

    Usage in train_grpo.py:

        her = HindsightReplayBuffer(k_hindsight=3)
        for rollout in rollouts:
            if rollout["f1"] < her.f1_threshold:
                hindsight = her.generate_hindsight_rollouts(
                    rollout, graph, ground_truth_path
                )
                rollouts.extend(hindsight)
    """

    def __init__(
        self,
        k_hindsight: int = 3,
        f1_threshold: float = 0.5,
        max_group_size: int = 16,
        seed: int = 42,
    ):
        self.k_hindsight = k_hindsight
        self.f1_threshold = f1_threshold
        self.max_group_size = max_group_size
        self._rng = np.random.default_rng(seed)
        self._stats = {
            "total_generated": 0,
            "total_source_episodes": 0,
            "avg_virtual_f1": 0.0,
        }

    @property
    def stats(self) -> Dict[str, Any]:
        return dict(self._stats)

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def generate_hindsight_rollouts(
        self,
        rollout: Dict[str, Any],
        graph: Any,  # TransactionGraph
        ground_truth_path: List[str],
        query_history: List[Dict[str, Any]],
        total_budget: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Generate K hindsight-relabeled rollouts from a single failed episode.

        Parameters
        ----------
        rollout : dict from run_rollout() with keys:
            prompts, completions, rewards, r_inv, scheme_type, f1, ...
        graph : TransactionGraph instance
        ground_truth_path : the real ground truth path (for reference)
        query_history : list of action dicts from the episode
        total_budget : episode budget

        Returns
        -------
        List of relabeled rollout dicts (same structure as input rollout).
        """
        # Only relabel failed episodes
        if rollout.get("f1", 1.0) >= self.f1_threshold:
            return []

        # Extract the entities the agent actually investigated
        evidence_chain = self._extract_evidence_chain(rollout)
        if len(evidence_chain) < 3:
            # Too few entities to form a meaningful virtual goal
            return []

        hindsight_rollouts: List[Dict[str, Any]] = []
        virtual_f1_scores: List[float] = []

        for k in range(self.k_hindsight):
            # Select a virtual ground truth from the agent's evidence
            virtual_gt = self._select_virtual_goal(
                evidence_chain, graph, k
            )
            if not virtual_gt or len(virtual_gt) < 2:
                continue

            # Recompute rewards with the virtual ground truth
            relabeled = self._recompute_rewards(
                rollout=rollout,
                virtual_ground_truth=virtual_gt,
                graph=graph,
                query_history=query_history,
                total_budget=total_budget,
            )

            if relabeled is not None:
                hindsight_rollouts.append(relabeled)
                virtual_f1_scores.append(relabeled["f1"])

        # Update stats
        if hindsight_rollouts:
            self._stats["total_generated"] += len(hindsight_rollouts)
            self._stats["total_source_episodes"] += 1
            # Rolling average of virtual F1
            n = self._stats["total_generated"]
            old_avg = self._stats["avg_virtual_f1"]
            new_avg = (old_avg * (n - len(virtual_f1_scores)) + sum(virtual_f1_scores)) / n
            self._stats["avg_virtual_f1"] = round(new_avg, 4)

        return hindsight_rollouts

    def should_relabel(self, rollout: Dict[str, Any]) -> bool:
        """Check if a rollout qualifies for HER relabeling."""
        return rollout.get("f1", 1.0) < self.f1_threshold

    # ------------------------------------------------------------------ #
    # Virtual goal selection                                               #
    # ------------------------------------------------------------------ #

    def _select_virtual_goal(
        self,
        evidence_chain: List[str],
        graph: Any,
        variant: int = 0,
    ) -> List[str]:
        """
        Select a connected subgraph from the evidence chain as virtual goal.

        Strategy: "future" — pick a random starting entity from the chain,
        then greedily extend by following edges in the transaction graph to
        other entities that are also in the evidence chain.

        Different `variant` values produce different virtual goals by
        starting from different entities.
        """
        if not evidence_chain:
            return []

        G = graph.graph  # networkx graph

        # Filter to entities that actually exist in the graph
        valid_entities = [e for e in evidence_chain if G.has_node(e)]
        if len(valid_entities) < 2:
            return valid_entities

        # Pick a starting entity (rotate based on variant for diversity)
        start_idx = (variant * 3 + 7) % len(valid_entities)
        start = valid_entities[start_idx]

        # BFS/greedy: extend from start following edges to other evidence entities
        connected: List[str] = [start]
        visited = {start}
        frontier = [start]

        while frontier:
            node = frontier.pop(0)
            # Check all neighbors in the transaction graph
            neighbors = set(G.successors(node)) | set(G.predecessors(node))
            for nbr in neighbors:
                if nbr in visited:
                    continue
                if nbr in set(valid_entities):
                    connected.append(nbr)
                    visited.add(nbr)
                    frontier.append(nbr)

        # Require at least 2 entities for a meaningful "scheme path"
        if len(connected) < 2:
            # Fallback: take a random contiguous slice of the evidence chain
            slice_len = min(len(valid_entities), max(3, len(valid_entities) // 2))
            start_pos = variant % max(len(valid_entities) - slice_len + 1, 1)
            connected = valid_entities[start_pos:start_pos + slice_len]

        return connected

    # ------------------------------------------------------------------ #
    # Reward recomputation                                                 #
    # ------------------------------------------------------------------ #

    def _recompute_rewards(
        self,
        rollout: Dict[str, Any],
        virtual_ground_truth: List[str],
        graph: Any,
        query_history: List[Dict[str, Any]],
        total_budget: int = 50,
    ) -> Optional[Dict[str, Any]]:
        """
        Create a relabeled copy of the rollout with rewards computed
        against the virtual ground truth.
        """
        evidence_chain = self._extract_evidence_chain(rollout)
        steps = rollout.get("steps", len(rollout.get("prompts", [])))
        compliance = rollout.get("compliance", 0.5)

        try:
            r_result = r_investigator(
                evidence_chain=evidence_chain,
                ground_truth_path=virtual_ground_truth,
                compliance_score=compliance,
                queries_used=steps,
                total_budget=total_budget,
                query_history=query_history,
                graph=graph,
                scheme_type=rollout.get("scheme_type", "hindsight"),
                is_codex_generated=False,
            )
        except Exception:
            return None

        r_inv = float(r_result["total"])
        f1 = float(r_result["detection_f1"])

        # Only keep if the virtual goal actually produces a better reward
        if f1 < 0.4:
            return None

        # Build relabeled rollout (same prompts/completions, new rewards)
        relabeled_rewards = list(rollout.get("rewards", []))
        if relabeled_rewards:
            relabeled_rewards[-1] = r_inv

        return {
            "prompts":        rollout["prompts"],
            "completions":    rollout["completions"],
            "rewards":        relabeled_rewards,
            "r_inv":          r_inv,
            "scheme_type":    rollout.get("scheme_type", "hindsight"),
            "f1":             f1,
            "f1_by_type":     {rollout.get("scheme_type", "hindsight"): f1},
            "steps":          steps,
            "compliance":     compliance,
            "action_entropy": rollout.get("action_entropy", 0.0),
            "first_action":   rollout.get("first_action", "none"),
            "action_counts":  rollout.get("action_counts", {}),
            "is_hindsight":   True,  # flag for logging
            "virtual_gt":     virtual_ground_truth,
        }

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _extract_evidence_chain(self, rollout: Dict[str, Any]) -> List[str]:
        """
        Extract the list of entities investigated from a rollout's completions.
        Parses ACTION:xxx ENTITY:yyy from each completion string.
        """
        import re
        evidence: List[str] = []
        seen: set = set()

        for completion in rollout.get("completions", []):
            # Parse entity from completion
            entity_match = re.search(r"ENTITY:(\S+)", completion)
            entities_match = re.search(r"ENTITIES:([\w,_\-]+)", completion)

            if entities_match:
                for e in entities_match.group(1).split(","):
                    e = e.strip()
                    if e and e not in seen:
                        evidence.append(e)
                        seen.add(e)
            elif entity_match:
                e = entity_match.group(1).strip()
                if e and e not in seen:
                    evidence.append(e)
                    seen.add(e)

        return evidence


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("HEIST Hindsight Experience Replay — Unit Test")
    print("=" * 70)

    # Create a mock failed rollout
    mock_rollout = {
        "prompts": [f"prompt_{i}" for i in range(10)],
        "completions": [
            "ACTION:query_transactions ENTITY:acc_100",
            "ACTION:query_transactions ENTITY:acc_200",
            "ACTION:trace_network ENTITY:acc_300",
            "ACTION:trace_network ENTITY:acc_400",
            "ACTION:trace_network ENTITY:acc_500",
            "ACTION:cross_reference_jurisdiction ENTITY:acc_600",
            "ACTION:cross_reference_jurisdiction ENTITY:acc_700",
            "ACTION:trace_network ENTITY:acc_800",
            "ACTION:trace_network ENTITY:acc_900",
            "ACTION:file_SAR ENTITIES:acc_100,acc_200,acc_300",
        ],
        "rewards": [0.0] * 9 + [0.15],
        "r_inv": 0.15,
        "scheme_type": "smurfing",
        "f1": 0.2,
        "steps": 10,
        "compliance": 0.3,
        "action_entropy": 1.2,
        "first_action": "query_transactions",
        "action_counts": {"query_transactions": 2, "trace_network": 4, "cross_reference_jurisdiction": 2, "file_SAR": 1},
    }

    her = HindsightReplayBuffer(k_hindsight=3, f1_threshold=0.5)

    # Test 1: should_relabel
    assert her.should_relabel(mock_rollout), "F1=0.2 should qualify for HER"
    assert not her.should_relabel({"f1": 0.8}), "F1=0.8 should NOT qualify"
    print("[1] ✓ should_relabel() correct")

    # Test 2: evidence extraction
    evidence = her._extract_evidence_chain(mock_rollout)
    assert len(evidence) == 9, f"Expected 9 entities, got {len(evidence)}"
    assert evidence[0] == "acc_100"
    print(f"[2] ✓ Extracted {len(evidence)} entities from completions")

    # Test 3: virtual goal selection (without real graph — uses fallback)
    class MockGraph:
        class G:
            def has_node(self, n): return True
            def successors(self, n): return []
            def predecessors(self, n): return []
        graph = G()

    goals = []
    for v in range(3):
        vg = her._select_virtual_goal(evidence, MockGraph(), variant=v)
        goals.append(vg)
        assert len(vg) >= 2, f"Virtual goal too small: {vg}"
    print(f"[3] ✓ Generated {len(goals)} virtual goals (sizes: {[len(g) for g in goals]})")

    # Test 4: stats tracking
    assert her.stats["total_generated"] == 0
    print("[4] ✓ Stats initialized correctly")

    print(f"\n{'='*70}")
    print("ALL HER TESTS PASSED")
    print(f"{'='*70}")
