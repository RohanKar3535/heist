"""
AdversarialCodex — domain-agnostic adversarial task generator for RL environments.

HEIST's contribution to the OpenEnv ecosystem:
    Any RL environment can import this class to get a self-evolving adversary
    that generates new task variants targeting the learning agent's weaknesses.

Usage (any OpenEnv environment):
    from adversarial_codex import AdversarialCodex

    codex = AdversarialCodex(
        env_description="financial crime investigation",
        scheme_types=["smurfing", "layering", "crypto_mixing"],
        llm_client=my_groq_or_gemini_client,
        novelty_threshold=0.05,
    )

    # After each episode:
    codex.update(scheme_type="smurfing", investigator_f1=0.25)

    # Every K episodes — generate a harder variant:
    new_task_code = codex.generate(target_weakness="smurfing")

    # At end of training — synthesize Zero-Day:
    zero_day = codex.synthesize_zero_day(top_n=3)

Design principles:
    1. Domain-agnostic: works for any env (cybersecurity, fraud, SRE, etc.)
    2. ELO-tracked: each task type has an ELO that rises when agent fails
    3. Novelty-gated: new tasks are rejected if structurally duplicate
    4. LLM-driven: generates executable task code at runtime, not offline
    5. Compositional Zero-Day: final attack engineered from failure modes

This file is intentionally standalone — no HEIST-specific imports.
Future OpenEnv environments can copy it as-is.
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# ELO utilities
# ---------------------------------------------------------------------------

def _elo_update(current: float, outcome: float, k: float = 32.0,
                reference: float = 1200.0) -> float:
    """
    Standard ELO update for a task against a fixed-reference agent.

    outcome: 1.0 = task won (agent failed), 0.0 = agent won, 0.5 = draw.
    Clamped to [800, 1600] — no task dominates permanently.
    """
    expected = 1.0 / (1.0 + 10.0 ** ((reference - current) / 400.0))
    return max(800.0, min(1600.0, current + k * (outcome - expected)))


def _softmax_weights(values: np.ndarray, temperature: float = 400.0) -> np.ndarray:
    """Numerically stable softmax with temperature over ELO-scale values."""
    v = values / max(temperature, 1e-8)
    v -= v.max()
    exp_v = np.exp(v)
    return exp_v / exp_v.sum()


# ---------------------------------------------------------------------------
# Novelty scoring
# ---------------------------------------------------------------------------

def _keyword_embedding(code: str, vocab: List[str]) -> np.ndarray:
    """
    Count occurrences of domain-specific keywords in code.
    Returns a normalized feature vector.

    Users should provide a domain-specific vocab. HEIST uses financial crime
    keywords (cash_deposit, wire_transfer, shell_company, etc.).
    """
    code_lower = code.lower()
    features = np.array([code_lower.count(kw) for kw in vocab], dtype=np.float64)
    norm = np.linalg.norm(features)
    if norm > 0:
        features /= norm
    return features


def compute_structural_novelty(
    new_code: str,
    existing_codes: List[str],
    vocab: List[str],
) -> float:
    """
    Novelty = minimum cosine distance to any existing task (nearest-neighbor).

    Using min instead of mean prevents the score from shrinking geometrically
    as the codex grows. Returns value in [0, 1]. Higher = more novel.
    """
    if not existing_codes or not vocab:
        return 1.0

    new_emb = _keyword_embedding(new_code, vocab)
    distances = []
    for code in existing_codes:
        old_emb = _keyword_embedding(code, vocab)
        cos_sim = float(np.dot(new_emb, old_emb))
        distances.append(1.0 - max(-1.0, min(1.0, cos_sim)))

    return round(float(np.min(distances)), 6)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class AdversarialCodex:
    """
    Self-evolving adversarial task generator for OpenEnv RL environments.

    The codex tracks which task types the agent struggles with (via ELO),
    generates new executable task variants using an LLM targeting those
    weaknesses, and gates new tasks with a structural novelty check to
    prevent duplicate injection.

    Parameters
    ----------
    env_description : str
        One-sentence description of the environment for LLM prompting.
        Example: "financial crime investigation"
    scheme_types : list of str
        Initial task/scheme type names.
        Example: ["smurfing", "layering", "crypto_mixing"]
    llm_call_fn : callable, optional
        Function that takes a prompt string and returns a response string.
        Signature: (prompt: str) -> Optional[str]
        If None, the codex uses deterministic fallback generation only.
    novelty_threshold : float
        Minimum structural novelty for a new task to enter the codex.
        Default 0.05 — rejects near-duplicate tasks.
    elo_k : float
        ELO K-factor. Higher = faster adaptation. Default 32.
    keyword_vocab : list of str, optional
        Domain-specific keywords for structural novelty scoring.
        If None, uses generic code tokens.
    """

    def __init__(
        self,
        env_description: str,
        scheme_types: List[str],
        llm_call_fn: Optional[Callable[[str], Optional[str]]] = None,
        novelty_threshold: float = 0.05,
        elo_k: float = 32.0,
        keyword_vocab: Optional[List[str]] = None,
    ) -> None:
        self.env_description = env_description
        self.scheme_types = list(scheme_types)
        self._llm_call = llm_call_fn
        self.novelty_threshold = novelty_threshold
        self.elo_k = elo_k

        # Per-task ELO: starts at 1200, rises when agent fails
        self.elo: Dict[str, float] = {s: 1200.0 for s in self.scheme_types}

        # Stored task codes for novelty comparison
        self.codex_codes: List[str] = []

        # Keyword vocabulary for structural novelty (domain-specific)
        self.vocab: List[str] = keyword_vocab or [
            "loop", "for ", "while", "if ", "random", "choice", "append",
            "edge", "node", "path", "amount", "transfer", "account",
        ]

        # History
        self._episode: int = 0
        self._generation_count: int = 0

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def update(
        self,
        scheme_type: str,
        investigator_f1: float,
        agent_won: Optional[bool] = None,
    ) -> None:
        """
        Record one episode result and update ELO.

        scheme_type    : the task type played this episode
        investigator_f1: agent F1 score [0, 1]
        agent_won      : override outcome (None = infer from F1 threshold)
        """
        self._episode += 1

        if scheme_type not in self.elo:
            self.elo[scheme_type] = 1200.0
            self.scheme_types.append(scheme_type)

        if agent_won is None:
            # Infer from F1: < 0.4 = task won, > 0.6 = agent won, else draw
            if investigator_f1 < 0.4:
                outcome = 1.0
            elif investigator_f1 > 0.6:
                outcome = 0.0
            else:
                outcome = 0.5
        else:
            outcome = 0.0 if agent_won else 1.0

        self.elo[scheme_type] = _elo_update(
            self.elo[scheme_type], outcome, self.elo_k
        )

    def select_task(self) -> str:
        """
        Sample a task type weighted by ELO.
        High-ELO (hard) tasks are sampled more — forces agent to improve
        on its hardest adversaries.
        """
        keys = list(self.elo.keys())
        elos = np.array([self.elo[k] for k in keys], dtype=np.float64)
        weights = _softmax_weights(elos, temperature=400.0)
        return str(np.random.choice(keys, p=weights))

    def generate(
        self,
        target_weakness: Optional[str] = None,
        validate_fn: Optional[Callable] = None,
    ) -> Tuple[Optional[str], Dict[str, Any]]:
        """
        Generate a new task variant targeting the weakest scheme type.

        target_weakness : override which scheme to target (default: highest ELO)
        validate_fn     : optional (code: str) -> bool for domain validation

        Returns (code: Optional[str], info: dict)
        """
        if target_weakness is None:
            target_weakness = self.top_by_elo(n=1)[0]

        info: Dict[str, Any] = {
            "target": target_weakness,
            "validated": False,
            "novelty": 0.0,
            "generation_count": self._generation_count,
        }

        # LLM generation
        code = None
        if self._llm_call is not None:
            prompt = self._build_prompt(target_weakness)
            response = self._llm_call(prompt)
            if response:
                code = self._extract_code(response)

        if code is None:
            info["error"] = "llm_unavailable_or_parse_failed"
            return None, info

        # Domain validation
        if validate_fn is not None:
            try:
                if not validate_fn(code):
                    info["error"] = "domain_validation_failed"
                    return None, info
            except Exception as e:
                info["error"] = f"validation_exception: {e}"
                return None, info

        # Novelty gate
        novelty = compute_structural_novelty(code, self.codex_codes, self.vocab)
        info["novelty"] = novelty

        if self.codex_codes and novelty < self.novelty_threshold:
            info["error"] = f"structural_duplicate (novelty={novelty:.3f} < {self.novelty_threshold})"
            return None, info

        # Accept into codex
        self.codex_codes.append(code)
        self._generation_count += 1
        info["validated"] = True
        info["generation_count"] = self._generation_count
        return code, info

    def synthesize_zero_day(
        self,
        top_n: int = 3,
        validate_fn: Optional[Callable] = None,
    ) -> Tuple[Optional[str], Dict[str, Any]]:
        """
        Synthesize a Zero-Day task by combining the evasion strategies of
        the top_n hardest tasks (by ELO). The result is provably hard by
        construction: if the agent fails on A, B, C individually, a task
        combining all three strategies will be even harder to detect.

        Returns (code: Optional[str], info: dict)
        """
        hard_schemes = self.top_by_elo(n=top_n)
        info: Dict[str, Any] = {
            "source_schemes": hard_schemes,
            "validated": False,
            "novelty": 0.0,
        }

        if self._llm_call is None:
            info["error"] = "no llm_call_fn provided"
            return None, info

        prompt = self._build_zero_day_prompt(hard_schemes)
        response = self._llm_call(prompt)

        if response:
            code = self._extract_code(response)
            if code:
                if validate_fn:
                    try:
                        if not validate_fn(code):
                            info["error"] = "zero_day_validation_failed"
                            return None, info
                    except Exception as e:
                        info["error"] = f"zero_day_exception: {e}"
                        return None, info

                novelty = compute_structural_novelty(code, self.codex_codes, self.vocab)
                self.codex_codes.append(code)
                info.update({"validated": True, "novelty": novelty, "code": code})
                return code, info

        info["error"] = "zero_day_generation_failed"
        return None, info

    def top_by_elo(self, n: int = 3) -> List[str]:
        """Return top-n hardest task types by ELO."""
        return sorted(self.elo, key=lambda s: -self.elo[s])[:n]

    def elo_table(self) -> str:
        """Human-readable ELO leaderboard."""
        rows = sorted(self.elo.items(), key=lambda x: -x[1])
        lines = [f"  AdversarialCodex ELO Table ({self.env_description}):"]
        for rank, (scheme, elo) in enumerate(rows, 1):
            bar = "█" * int((elo - 800) / 40)
            lines.append(f"  {rank:2d}. {scheme:<24s} {elo:7.1f}  {bar}")
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self.codex_codes)

    def __repr__(self) -> str:
        return (
            f"AdversarialCodex(env={self.env_description!r}, "
            f"n_tasks={len(self.scheme_types)}, "
            f"codex_size={len(self.codex_codes)}, "
            f"episodes={self._episode})"
        )

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _build_prompt(self, target: str) -> str:
        return f"""\
You are an adversarial task designer for a {self.env_description} environment.

The learning agent is weakest at handling: {target}
Generate a new Python function that creates a harder variant of this task type.
The function should be executable, structurally novel, and harder for the agent.

Respond with ONLY the Python function code. No explanation. No markdown fencing.
"""

    def _build_zero_day_prompt(self, hard_schemes: List[str]) -> str:
        schemes_str = "\n".join(f"- {s}" for s in hard_schemes)
        return f"""\
You are an adversarial task designer for a {self.env_description} environment.

The agent struggles most with these task types:
{schemes_str}

Synthesize a single Python function that COMBINES the difficulty elements of all
{len(hard_schemes)} task types into one coordinated attack. This Zero-Day should
be provably harder than any individual task type.

Respond with ONLY the Python function code. No explanation. No markdown fencing.
"""

    def _extract_code(self, response: str) -> Optional[str]:
        code = response.strip()
        code = re.sub(r'^```python\s*', '', code)
        code = re.sub(r'^```\s*', '', code)
        code = re.sub(r'\s*```$', '', code)
        return code if code.strip().startswith('def') else None
