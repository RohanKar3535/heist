"""
Oversight Agent — Step 10.

Monitors the Investigator, Criminal, and Compliance Expert in real-time.
Scans for 6 specific anomaly flags and logs them to the episode_log.

Flag 1: Investigator queried same entity 3+ times (warning)
Flag 2: Criminal reusing scheme structure with cosine similarity > 0.85 (warning)
Flag 3: Investigator filed SAR with < 3 entities in evidence chain (critical)
Flag 4: Compliance expert preferences ignored (query count > 50% over limit) (warning)
Flag 5: Criminal morphed > 1 time in episode (critical)
Flag 6: Investigator skipped CrossReference phase entirely (critical)

Computes R_oversight = anomalies_caught / anomalies_total.
"""

from __future__ import annotations

import collections
import inspect
import sys
import os
from typing import Any, Dict, List, Optional

import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

try:
    from agents.criminal import _scheme_embedding
except ImportError:
    # Fallback if cannot import
    def _scheme_embedding(scheme_code: str) -> np.ndarray:
        ngrams: Dict[str, int] = {}
        code = scheme_code.lower()
        for i in range(len(code) - 2):
            ng = code[i:i+3]
            ngrams[ng] = ngrams.get(ng, 0) + 1
        vocab = [
            "gra", "pha", "edd", "dge", "nod", "ode", "amo", "mou", "unt",
            "jur", "uri", "isd", "dic", "ict", "tio", "ion", "tra", "ran",
            "ans", "nsa", "sac", "act", "she", "hel", "ell", "cry", "ryp",
            "ypt", "pto", "smu", "mur", "urf", "lay", "aye", "yer", "wir",
            "ire", "cas", "ash", "dep", "epo", "pos", "sit", "swi", "wif",
            "ift", "haw", "awa", "wal", "ala", "off", "ffs", "fsh", "sho",
            "hor", "ore", "pat", "ath", "loo", "oop", "hop", "mul", "ule",
            "spl", "pli", "lit", "fan", "mix", "ixe", "tur", "umb", "mbl",
            "ble", "inv", "nvo", "voi", "oic", "ice", "pri", "ric", "rng",
            "for", "ran", "dom", "int", "flo", "loa", "uni", "nif", "ifo",
        ]
        vec = np.array([ngrams.get(v, 0) for v in vocab], dtype=np.float64)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec


# ---------------------------------------------------------------------------
# Oversight Agent
# ---------------------------------------------------------------------------

class OversightAgent:
    def __init__(self):
        self.flags_raised: List[Dict[str, Any]] = []

    def _add_flag(
        self,
        episode_log: List[dict],
        step: int,
        flag_id: int,
        description: str,
        severity: str
    ) -> None:
        """Appends a flag to the episode_log and tracks it."""
        prefix = "🚨" if severity == "critical" else "⚠️"
        flag_entry = {
            "event": "oversight_flag",
            "step": step,
            "flag_id": flag_id,
            "severity": severity,
            "description": f"{prefix} Flag {flag_id}: {description}",
        }
        episode_log.append(flag_entry)
        self.flags_raised.append(flag_entry)

    def monitor_episode(
        self,
        state: Any,
        episode_log: List[dict],
        episode_number: int,
        new_criminal_code: Optional[str] = None,
        existing_codex_codes: Optional[List[str]] = None,
    ) -> None:
        """
        Analyze the full episode state and append flags to the episode_log inline.
        Call this at the end of an episode or during steps to detect issues.
        """
        history = getattr(state, "query_history", [])
        step = getattr(state, "step_count", 0)

        # We keep track of which flags we've already raised this episode
        # so we don't spam the log with identical warnings. 
        # For this simple agent, we analyze history and emit flags as needed.
        raised_ids = {f["flag_id"] for f in self.flags_raised}

        # Flag 1: Investigator queried same entity 3+ times (warning)
        if 1 not in raised_ids:
            query_actions = [h for h in history if h.get("action_type") == "query_transactions"]
            counts = collections.Counter(h.get("params", {}).get("entity_id") for h in query_actions)
            for entity, count in counts.items():
                if count >= 3 and entity:
                    self._add_flag(
                        episode_log, step, 1,
                        f"Investigator queried {entity} {count} times (wasteful)",
                        "warning"
                    )
                    break

        # Flag 2: Criminal reusing scheme structure with cosine similarity > 0.85 (warning)
        if 2 not in raised_ids and new_criminal_code and existing_codex_codes:
            new_emb = _scheme_embedding(new_criminal_code)
            max_sim = 0.0
            for old_code in existing_codex_codes:
                old_emb = _scheme_embedding(old_code)
                sim = float(np.dot(new_emb, old_emb))
                max_sim = max(max_sim, sim)
            
            if max_sim > 0.85:
                self._add_flag(
                    episode_log, step, 2,
                    f"Criminal scheme highly similar to existing codex entry (sim={max_sim:.2f})",
                    "warning"
                )

        # Flag 3: Investigator filed SAR with < 3 entities in evidence chain (critical)
        if 3 not in raised_ids:
            sar_actions = [h for h in history if h.get("action_type") == "file_SAR"]
            if sar_actions:
                chain = sar_actions[0].get("params", {}).get("evidence_chain", [])
                if len(chain) < 3:
                    self._add_flag(
                        episode_log, step, 3,
                        f"SAR filed with insufficient evidence chain ({len(chain)} entities)",
                        "critical"
                    )

        # Flag 4: Compliance expert preferences ignored by > 50% (warning)
        # Phase 1: no strict limit, Phase 2 expected max ~20, Phase 3 expected max ~10.
        if 4 not in raised_ids:
            expected_limit = 50
            if 20 <= episode_number < 50:
                expected_limit = 20
            elif episode_number >= 50:
                expected_limit = 10
                
            query_count = sum(1 for h in history if h.get("action_type") == "query_transactions" or h.get("action_type") == "trace_network")
            # If greater than 1.5 * limit
            if query_count > (expected_limit * 1.5):
                self._add_flag(
                    episode_log, step, 4,
                    f"Expert preferences ignored: Q-count {query_count} exceeds expected {expected_limit} by >50%",
                    "warning"
                )

        # Flag 5: Criminal morphed > 1 time in episode (critical)
        if 5 not in raised_ids:
            morph_count = getattr(state, "morph_count", 0)
            if morph_count > 1:
                self._add_flag(
                    episode_log, step, 5,
                    f"Criminal violated stackelberg rules: morphed {morph_count} times",
                    "critical"
                )

        # Flag 6: Investigator skipped CrossReference phase entirely (critical)
        if 6 not in raised_ids:
            sar_actions = [h for h in history if h.get("action_type") == "file_SAR"]
            if sar_actions:
                has_xref = any(h.get("action_type") == "cross_reference_jurisdiction" for h in history)
                if not has_xref:
                    self._add_flag(
                        episode_log, step, 6,
                        "Investigator filed SAR without CrossReference queries",
                        "critical"
                    )

    def compute_reward(self, actual_anomalies_present: List[int]) -> float:
        """
        R_oversight = anomalies_caught / anomalies_total
        Takes actual_anomalies_present (e.g. [1, 3, 6]) that occurred in the episode.
        """
        if not actual_anomalies_present:
            return 1.0  # Perfect if nothing to catch and none caught (or 0 if penalizing false positives, but requirement says anomalies_caught / anomalies_total)
            
        caught = sum(1 for f in self.flags_raised if f["flag_id"] in actual_anomalies_present)
        return float(caught) / len(actual_anomalies_present)


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("HEIST Oversight Agent — Smoke Test")
    print("=" * 70)

    class DummyState:
        def __init__(self, history, morphs, step):
            self.query_history = history
            self.morph_count = morphs
            self.step_count = step

    history = [
        {"action_type": "query_transactions", "params": {"entity_id": "acc_1"}},
        {"action_type": "query_transactions", "params": {"entity_id": "acc_1"}},
        {"action_type": "query_transactions", "params": {"entity_id": "acc_1"}}, # Flag 1 (3 times)
        {"action_type": "trace_network", "params": {"entity_id": "acc_2"}},
        {"action_type": "trace_network", "params": {"entity_id": "acc_3"}},
        {"action_type": "trace_network", "params": {"entity_id": "acc_4"}},
        # Phase 3 limit is 10 queries. Above we have 6. Let's add 10 more to violate by 50%
    ] + [{"action_type": "trace_network", "params": {"entity_id": f"acc_d_{i}"}} for i in range(10)]
    
    # Still no cross_reference. Let's file SAR with short chain.
    history.append({
        "action_type": "file_SAR", 
        "params": {"evidence_chain": [{"entity_id": "acc_1"}, {"entity_id": "acc_2"}]} # Flag 3 (< 3 entities), Flag 6 (skipped xref)
    })

    log = []
    state = DummyState(history, morphs=2, step=17) # Flag 5 (morphs > 1)

    code_a = "def inject(graph): pass"
    code_b = "def inject(graph): pass  # highly similar"
    
    agent = OversightAgent()
    agent.monitor_episode(
        state=state, 
        episode_log=log, 
        episode_number=60, # Phase 3 => Expected 10. We have 16 queries. > 15 = Flag 4
        new_criminal_code=code_b,
        existing_codex_codes=[code_a] # Flag 2 (> 0.85 similarity)
    )

    print("\n[1] Verifying flagged anomalies in episode_log...")
    detected_ids = set()
    for entry in log:
        if entry.get("event") == "oversight_flag":
            detected_ids.add(entry["flag_id"])
            print(f"  {entry['description']}")

    # Check all 6 flags
    assert len(detected_ids) == 6, f"Expected 6 flags, got {len(detected_ids)}"
    for i in range(1, 7):
        assert i in detected_ids, f"Flag {i} missing!"
    print("  > All 6 deliberate violations successfully caught!")

    print("\n[2] Testing R_oversight calculation...")
    actual = [1, 2, 3, 4, 5, 6]
    r = agent.compute_reward(actual)
    assert r == 1.0, f"R_oversight should be 1.0, got {r}"
    print(f"  R_oversight = {r:.2f} (caught {len(detected_ids)}/{len(actual)})")
    
    # What if only 3 actual existed?
    r_partial = agent.compute_reward([1, 2, 3])
    # The agent might have flagged false positives, but the formula requested is caught / total.
    # Caught = 3, Total = 3 => 1.0. If we had 4 actual and caught 3 => 0.75.
    
    # Re-run with clean state
    agent2 = OversightAgent()
    clean_history = [
        {"action_type": "query_transactions", "params": {"entity_id": "acc_1"}},
        {"action_type": "cross_reference_jurisdiction", "params": {"entity_id": "acc_1"}},
        {"action_type": "file_SAR", "params": {"evidence_chain": [{"entity_id": "acc_1"}, {"entity_id": "acc_2"}, {"entity_id": "acc_3"}]}}
    ]
    clean_state = DummyState(clean_history, morphs=0, step=3)
    clean_log = []
    agent2.monitor_episode(clean_state, clean_log, 60)
    
    clean_flags = [e for e in clean_log if e.get("event") == "oversight_flag"]
    assert len(clean_flags) == 0, f"Expected 0 false positives, got {len(clean_flags)}"
    print("\n[3] Clean episode verification...")
    print("  > No false positives on clean episode.")

    print(f"\n{'='*70}")
    print(f"ALL OVERSIGHT TESTS PASSED")
    print(f"{'='*70}")
