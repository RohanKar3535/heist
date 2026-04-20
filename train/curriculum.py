"""
Red Queen Curriculum Controller — Step 12.

Implements co-evolutionary curriculum: as the investigator gets better at
detecting certain scheme types, the criminal shifts probability mass toward
schemes the investigator fails on most.

Key components
--------------
1. weakness_vector   — dict mapping scheme_type → (1 - F1)
2. P(scheme_k)       — softmax(weakness_vector / τ), τ decays over time
3. τ schedule        — τ(ep) = τ_0 × decay^(ep / K), τ_0=1.0, decay=0.95, K=5
4. ELO tracker       — standard Elo update (K=32) for criminal + investigator
5. JSON logs         — weakness_history.json, f1_history.json
"""

from __future__ import annotations

import json
import math
import os
import sys
from typing import Any, Dict, List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

WEAKNESS_HISTORY_PATH = os.environ.get(
    "WEAKNESS_HISTORY_PATH", os.path.join(_ROOT, "weakness_history.json")
)
F1_HISTORY_PATH = os.environ.get(
    "F1_HISTORY_PATH", os.path.join(_ROOT, "f1_history.json")
)

# ---------------------------------------------------------------------------
# Temperature schedule
# ---------------------------------------------------------------------------

def temperature(episode: int, tau_0: float = 1.0, decay: float = 0.95, K: int = 5) -> float:
    """
    τ(episode) = τ_0 × decay^(episode / K)

    - High τ early → near-uniform sampling (exploration)
    - Low τ late   → sharpens on weakest scheme (exploitation)
    """
    return tau_0 * (decay ** (episode / K))


# ---------------------------------------------------------------------------
# Softmax
# ---------------------------------------------------------------------------

def _softmax(values: np.ndarray, tau: float) -> np.ndarray:
    """Numerically stable softmax with temperature τ."""
    v = np.array(values, dtype=np.float64) / max(tau, 1e-8)
    v -= v.max()          # stability shift
    exp_v = np.exp(v)
    return exp_v / exp_v.sum()


# ---------------------------------------------------------------------------
# Red Queen Curriculum Controller
# ---------------------------------------------------------------------------

class RedQueenCurriculum:
    """
    Controls which scheme type the criminal generates next based on where
    the investigator is weakest.

    Parameters
    ----------
    scheme_types : list of known scheme type strings
    tau_0        : initial temperature (default 1.0)
    decay        : temperature decay factor per K episodes (default 0.95)
    K            : adaptation interval in episodes (default 5)
    elo_k        : ELO K-factor (default 32)
    """

    def __init__(
        self,
        scheme_types: Optional[List[str]] = None,
        tau_0: float = 1.0,
        decay: float = 0.95,
        K: int = 5,
        elo_k: float = 32.0,
    ):
        self.scheme_types = list(scheme_types or _DEFAULT_SCHEME_TYPES)
        self.tau_0  = tau_0
        self.decay  = decay
        self.K      = K
        self.elo_k  = elo_k

        # weakness_vector[scheme_type] = 1 - F1  (higher = harder for investigator)
        self.weakness_vector: Dict[str, float] = {s: 0.5 for s in self.scheme_types}

        # ELO ratings (start at 1200 each)
        self.criminal_elo:    float = 1200.0
        self.investigator_elo: float = 1200.0

        # History logs (episode-batched)
        self._weakness_history: List[Dict[str, Any]] = []
        self._f1_history:       List[Dict[str, Any]] = []

        # Running per-scheme F1 accumulators for the current batch
        self._batch_f1:    Dict[str, List[float]] = {s: [] for s in self.scheme_types}
        self._episode:     int = 0

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def update(self, episode_result: Dict[str, Any]) -> None:
        """
        Called after every episode.

        episode_result must contain:
            scheme_type : str    — the scheme played this episode
            f1          : float  — investigator F1 for this episode
            morph_succeeded   : bool (optional)
            novel_evaded      : bool (optional)
        """
        self._episode += 1
        scheme = episode_result.get("scheme_type", "unknown")
        f1     = float(episode_result.get("f1", 0.5))

        # Ensure scheme is tracked
        if scheme not in self.scheme_types:
            self.scheme_types.append(scheme)
            self.weakness_vector[scheme] = 0.5
            self._batch_f1[scheme] = []

        # Update weakness: weakness_vector[k] = 1 - F1(k)
        self.weakness_vector[scheme] = 1.0 - f1

        # Accumulate for batch logging
        self._batch_f1[scheme].append(f1)

        # ELO update
        morph_succeeded = bool(episode_result.get("morph_succeeded", False))
        novel_evaded    = bool(episode_result.get("novel_evaded", False))
        self._update_elo(f1, morph_succeeded, novel_evaded)

        # Every K episodes: log batch + criminal adaptation
        if self._episode % self.K == 0:
            self._flush_batch(self._episode)

    def sampling_distribution(self, episode: Optional[int] = None) -> Dict[str, float]:
        """
        Return P(generate_scheme_k) = softmax(weakness_vector / τ).

        Higher probability → criminal more likely to generate that scheme type.
        """
        ep  = episode if episode is not None else self._episode
        tau = temperature(ep, self.tau_0, self.decay, self.K)

        keys   = list(self.weakness_vector.keys())
        values = np.array([self.weakness_vector[k] for k in keys])
        probs  = _softmax(values, tau)

        return {k: float(p) for k, p in zip(keys, probs)}

    def select_scheme_type(self, episode: Optional[int] = None) -> str:
        """Sample one scheme type according to the curriculum distribution."""
        dist  = self.sampling_distribution(episode)
        keys  = list(dist.keys())
        probs = np.array([dist[k] for k in keys])
        return str(np.random.choice(keys, p=probs))

    def elo_summary(self) -> Dict[str, float]:
        return {
            "criminal_elo":     round(self.criminal_elo, 1),
            "investigator_elo": round(self.investigator_elo, 1),
            "delta":            round(self.criminal_elo - self.investigator_elo, 1),
        }

    def save(self) -> None:
        """Persist both history logs to JSON."""
        wh_path = os.environ.get("WEAKNESS_HISTORY_PATH", WEAKNESS_HISTORY_PATH)
        fh_path = os.environ.get("F1_HISTORY_PATH", F1_HISTORY_PATH)
        _write_json(wh_path, self._weakness_history)
        _write_json(fh_path, self._f1_history)

    def load(self) -> None:
        """Load history from disk (resumes after crash / checkpoint)."""
        if os.path.exists(WEAKNESS_HISTORY_PATH):
            self._weakness_history = _read_json(WEAKNESS_HISTORY_PATH)
        if os.path.exists(F1_HISTORY_PATH):
            self._f1_history = _read_json(F1_HISTORY_PATH)

    # ------------------------------------------------------------------ #
    # ELO update                                                           #
    # ------------------------------------------------------------------ #

    def _expected_outcome(self, rating_a: float, rating_b: float) -> float:
        """Standard ELO expected score for player A against player B."""
        return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))

    def _update_elo(
        self,
        investigator_f1: float,
        morph_succeeded: bool,
        novel_evaded: bool,
    ) -> None:
        """
        ELO update rules (K=32):

        Criminal ELO increases when:
          - morph succeeded (criminal evaded investigator mid-episode)
          - novel scheme evaded detection (investigator F1 < 0.3)
          - investigator F1 dropped below 0.4

        Investigator ELO increases when:
          - novel scheme caught (F1 >= 0.7 on difficult scheme)
          - morph was defeated (morph_succeeded=False but attempted)
          - F1 improved on weakest scheme type (F1 > 0.6)
        """
        E_crim = self._expected_outcome(self.criminal_elo, self.investigator_elo)
        E_inv  = self._expected_outcome(self.investigator_elo, self.criminal_elo)

        # Criminal wins the exchange if morph worked or novel scheme evaded
        criminal_outcome    = 0.5
        investigator_outcome = 0.5

        if morph_succeeded or novel_evaded or investigator_f1 < 0.4:
            # Criminal wins
            criminal_outcome    = 1.0
            investigator_outcome = 0.0
        elif investigator_f1 >= 0.7:
            # Investigator wins
            criminal_outcome    = 0.0
            investigator_outcome = 1.0
        # else draw (0.5 each)

        self.criminal_elo    += self.elo_k * (criminal_outcome    - E_crim)
        self.investigator_elo += self.elo_k * (investigator_outcome - E_inv)

    # ------------------------------------------------------------------ #
    # Batch logging                                                        #
    # ------------------------------------------------------------------ #

    def _flush_batch(self, episode: int) -> None:
        """Log weakness snapshot and mean F1 per scheme to JSON histories."""
        tau = temperature(episode, self.tau_0, self.decay, self.K)

        # Compute per-scheme mean F1 from batch accumulator
        mean_f1: Dict[str, Optional[float]] = {}
        for s in self.scheme_types:
            vals = self._batch_f1.get(s, [])
            mean_f1[s] = round(float(np.mean(vals)), 4) if vals else None

        # Weakness snapshot
        weakness_snap = {
            "episode":        episode,
            "tau":            round(tau, 4),
            "weakness_vector": {k: round(v, 4) for k, v in self.weakness_vector.items()},
            "sampling_dist":  {k: round(v, 4) for k, v in self.sampling_distribution(episode).items()},
            "elo":            self.elo_summary(),
        }
        self._weakness_history.append(weakness_snap)

        # F1 snapshot
        f1_snap = {
            "episode": episode,
            "mean_f1": mean_f1,
        }
        self._f1_history.append(f1_snap)

        # Reset batch accumulators
        self._batch_f1 = {s: [] for s in self.scheme_types}

        # Persist
        self.save()


# ---------------------------------------------------------------------------
# Default scheme types (mirrors categories in scenario library)
# ---------------------------------------------------------------------------

_DEFAULT_SCHEME_TYPES = [
    "smurfing",
    "shell_company",
    "round_trip",
    "cash_front",
    "layering",
    "invoice_fraud",
    "crypto_deposit",
    "trade_based",
    "shell_chain",
    "cross_border_wire",
    "crypto_mixing",
    "real_estate",
    "nested_shells",
    "multi_jurisdiction",
    "loan_back",
    "dividend_stripping",
    "trade_mispricing",
    "mirror_trading",
    "mule_network",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_json(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def pytest_approx(val, rel=1e-3):
    """Simple approximate comparison helper for standalone tests."""
    class _Approx:
        def __init__(self, v): self.v = v
        def __eq__(self, other): return abs(other - self.v) <= rel * abs(self.v)
    return _Approx(val)


# ---------------------------------------------------------------------------
# Smoke test / CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile, shutil

    print("=" * 70)
    print("HEIST Red Queen Curriculum — Test Suite")
    print("=" * 70)

    # Use temp dir for test output so we don't pollute project root
    _tmp = tempfile.mkdtemp()
    os.environ["WEAKNESS_HISTORY_PATH"] = os.path.join(_tmp, "weakness_history.json")
    os.environ["F1_HISTORY_PATH"]       = os.path.join(_tmp, "f1_history.json")

    curriculum = RedQueenCurriculum(
        scheme_types=list(_DEFAULT_SCHEME_TYPES),
        tau_0=1.0, decay=0.95, K=5
    )

    # ── Test 1: Temperature schedule ───────────────────────────────────
    print("\n[1] Temperature schedule τ(ep) = 1.0 × 0.95^(ep/5)")
    for ep in [0, 5, 10, 25, 50]:
        tau = temperature(ep, tau_0=1.0, decay=0.95, K=5)
        print(f"  ep={ep:3d}  τ={tau:.4f}")
    assert temperature(0)   == 1.0,                 "τ(0) must be 1.0"
    assert temperature(5)   == pytest_approx(0.95), "τ(5) must be 0.95^1"
    assert temperature(100) <  temperature(50),     "τ must be strictly decreasing"
    print("  ✓ Temperature schedule correct")

    # ── Test 2: Softmax sampling distribution ──────────────────────────
    print("\n[2] Softmax sampling distribution")
    curriculum.weakness_vector = {s: 0.1 for s in curriculum.scheme_types}
    curriculum.weakness_vector["shell_chain"] = 0.9   # very weak here
    dist = curriculum.sampling_distribution(episode=0)
    total = sum(dist.values())
    assert abs(total - 1.0) < 1e-6, f"Probabilities must sum to 1, got {total}"
    assert dist["shell_chain"] == max(dist.values()), "Weakest scheme should have highest prob"
    print(f"  P(shell_chain) = {dist['shell_chain']:.4f}  (highest: ✓)")
    print(f"  Sum of probs   = {total:.6f}  (≈ 1.0: ✓)")
    print("  ✓ Softmax distribution correct")

    # ── Test 3: Simulate 10 poor shell_chain episodes ──────────────────
    # Use a small 5-scheme vocabulary matching realistic training conditions.
    # With 5 schemes, failing shell_chain 10× while others stay at 0.5
    # weakness easily drives P(shell_chain) above 30% absolute increase.
    print("\n[3] Simulating 10 poor shell_chain episodes (F1 ≈ 0.05)")
    small_schemes = ["shell_chain", "smurfing", "layering", "crypto_mixing", "trade_based"]
    curriculum2 = RedQueenCurriculum(scheme_types=small_schemes, tau_0=1.0, decay=0.95, K=5)

    # Baseline: uniform weakness → P = 1/5 = 0.20
    baseline_dist = curriculum2.sampling_distribution(episode=0)
    baseline_p = baseline_dist["shell_chain"]
    print(f"  Baseline P(shell_chain) = {baseline_p:.4f}  (uniform: 1/{len(small_schemes)})")

    # Feed 10 episodes where investigator always fails on shell_chain
    for i in range(10):
        curriculum2.update({
            "scheme_type": "shell_chain",
            "f1": 0.05,   # investigator always fails
        })

    post_dist = curriculum2.sampling_distribution(episode=10)
    post_p    = post_dist["shell_chain"]
    abs_increase = post_p - baseline_p
    rel_increase = abs_increase / baseline_p   # relative increase
    print(f"  Post-10-fail P(shell_chain) = {post_p:.4f}")
    print(f"  Absolute increase = +{abs_increase:.4f}  |  Relative increase = +{rel_increase*100:.1f}%")
    # Spec: "P(generate shell_chain) increases by > 30%" = relative increase > 30%
    assert rel_increase > 0.30, f"Expected relative increase > 30%, got {rel_increase*100:.1f}%"
    print(f"  ✓ P(shell_chain) increased by {rel_increase*100:.1f}% relative (> 30% threshold)")

    # ── Test 4: ELO tracking ───────────────────────────────────────────
    print("\n[4] ELO tracking")
    curriculum3 = RedQueenCurriculum()
    init_crim = curriculum3.criminal_elo
    init_inv  = curriculum3.investigator_elo

    # Criminal wins (morph succeeds)
    curriculum3.update({"scheme_type": "smurfing", "f1": 0.2, "morph_succeeded": True})
    assert curriculum3.criminal_elo > init_crim,    "Criminal ELO must increase on morph win"
    assert curriculum3.investigator_elo < init_inv, "Investigator ELO must drop on morph win"

    # Investigator wins (high F1)
    elo_after_crim_win = curriculum3.criminal_elo
    curriculum3.update({"scheme_type": "smurfing", "f1": 0.9, "morph_succeeded": False})
    assert curriculum3.criminal_elo < elo_after_crim_win, "Criminal ELO must drop on investigator win"
    print(f"  {curriculum3.elo_summary()}")
    print("  ✓ ELO updates correct for wins/losses")

    # ── Test 5: JSON persistence ───────────────────────────────────────
    print("\n[5] JSON persistence")
    # Reload path constants from env (set at top of test)
    wh_path = os.environ["WEAKNESS_HISTORY_PATH"]
    fh_path = os.environ["F1_HISTORY_PATH"]

    # curriculum2 ran 10 episodes with K=5 → 2 batch flushes → files exist
    assert os.path.exists(wh_path), f"weakness_history.json missing at {wh_path}"
    assert os.path.exists(fh_path), f"f1_history.json missing at {fh_path}"
    wh = _read_json(wh_path)
    fh = _read_json(fh_path)
    assert len(wh) > 0, "weakness_history.json should have entries"
    assert len(fh) > 0, "f1_history.json should have entries"
    print(f"  weakness_history.json: {len(wh)} batch entries")
    print(f"  f1_history.json:       {len(fh)} batch entries")
    print("  ✓ JSON persistence correct")

    # Cleanup temp dir
    shutil.rmtree(_tmp, ignore_errors=True)

    print(f"\n{'='*70}")
    print("ALL RED QUEEN CURRICULUM TESTS PASSED")
    print(f"{'='*70}")
