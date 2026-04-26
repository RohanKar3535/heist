"""
Investigation Skill Discovery — Step 18.

Discovers reusable investigation patterns (skills) from successful episodes
via frequent subsequence mining, then injects them into the agent's system
prompt so it can leverage learned strategies.

Inspired by the Options Framework (Sutton et al., 1999) adapted for LLM
tool-use agents.  Instead of learning option policies, we mine for recurring
action subsequences and surface them as natural-language investigation
strategies that the LLM can follow.

Core insight
------------
Across successful episodes (F1 > 0.6), the trained investigator converges
on specific action patterns:
    - "query → query → trace → trace → trace" (deep triage-to-trace)
    - "trace → cross_ref → trace → cross_ref" (jurisdiction sweep)
    - "query → trace → lookup → trace → cross_ref → SAR" (full pipeline)

These patterns are *skills* — reusable investigation strategies that work
across different scheme types.  Surfacing them in the prompt lets the agent:
    1. Execute proven strategies faster (fewer wasted actions)
    2. Transfer strategies across scheme types
    3. Provide a reusable skill library for other tool-augmented RL projects

Parameters
----------
min_support : float  — minimum fraction of successful episodes containing
                       the subsequence for it to qualify as a skill (default 0.6)
min_length  : int    — minimum action sequence length for a skill (default 3)
max_skills  : int    — maximum number of skills to keep (default 8)
mine_every  : int    — mine for skills every N episodes (default 10)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

SKILLS_PATH = os.environ.get(
    "SKILLS_PATH", os.path.join(_ROOT, "skills_library.json")
)


# ---------------------------------------------------------------------------
# Skill dataclass
# ---------------------------------------------------------------------------

@dataclass
class InvestigationSkill:
    """A discovered reusable investigation pattern."""
    name: str                          # human-readable name
    action_sequence: List[str]         # ordered list of action types
    support: float                     # fraction of successful episodes containing this
    avg_f1: float                      # mean F1 of episodes where this skill appeared
    discovery_episode: int             # episode when this skill was first discovered
    usage_count: int = 0               # how many times the agent has used this
    description: str = ""              # natural-language description

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "InvestigationSkill":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Skill Library
# ---------------------------------------------------------------------------

class SkillLibrary:
    """
    Discovers, stores, and surfaces investigation skills.

    Usage in train_grpo.py:

        skill_lib = SkillLibrary()

        # After each episode:
        skill_lib.record_episode(action_type_log, f1, scheme_type)

        # Every 10 episodes:
        new_skills = skill_lib.mine_skills(current_episode=ep)

        # Inject into system prompt:
        extra_prompt = skill_lib.get_skill_prompt_injection()
    """

    def __init__(
        self,
        min_support: float = 0.6,
        min_length: int = 3,
        max_skills: int = 8,
        mine_every: int = 10,
        f1_success_threshold: float = 0.6,
    ):
        self.min_support = min_support
        self.min_length = min_length
        self.max_skills = max_skills
        self.mine_every = mine_every
        self.f1_success_threshold = f1_success_threshold

        self.skills: List[InvestigationSkill] = []
        self._episode_log: List[Dict[str, Any]] = []  # raw episode recordings
        self._episode_count: int = 0

    # ------------------------------------------------------------------ #
    # Recording                                                            #
    # ------------------------------------------------------------------ #

    def record_episode(
        self,
        action_sequence: List[str],
        f1: float,
        scheme_type: str = "",
    ) -> None:
        """Record an episode's action sequence for later mining."""
        self._episode_count += 1
        self._episode_log.append({
            "episode": self._episode_count,
            "actions": list(action_sequence),
            "f1": f1,
            "scheme_type": scheme_type,
        })

    # ------------------------------------------------------------------ #
    # Mining                                                               #
    # ------------------------------------------------------------------ #

    def mine_skills(self, current_episode: int = 0) -> List[InvestigationSkill]:
        """
        Run frequent subsequence mining on successful episodes.

        Returns list of newly discovered skills (may be empty).
        """
        # Filter to successful episodes only
        successful = [
            ep for ep in self._episode_log
            if ep["f1"] >= self.f1_success_threshold
        ]

        if len(successful) < 3:
            # Too few successful episodes to mine meaningfully
            return []

        # Extract all subsequences of length min_length to min_length+3
        subseq_counts: Dict[Tuple[str, ...], List[float]] = {}

        for ep in successful:
            actions = ep["actions"]
            seen_in_episode: Set[Tuple[str, ...]] = set()

            for length in range(self.min_length, min(self.min_length + 4, len(actions) + 1)):
                for start in range(len(actions) - length + 1):
                    subseq = tuple(actions[start:start + length])
                    if subseq not in seen_in_episode:
                        seen_in_episode.add(subseq)
                        if subseq not in subseq_counts:
                            subseq_counts[subseq] = []
                        subseq_counts[subseq].append(ep["f1"])

        # Filter by support threshold
        n_successful = len(successful)
        candidates: List[Tuple[Tuple[str, ...], float, float]] = []

        for subseq, f1_list in subseq_counts.items():
            support = len(f1_list) / n_successful
            if support >= self.min_support:
                avg_f1 = float(np.mean(f1_list))
                candidates.append((subseq, support, avg_f1))

        # Sort by (support * avg_f1) descending — best skills first
        candidates.sort(key=lambda x: x[1] * x[2], reverse=True)

        # Deduplicate: remove subsequences that are subsets of longer ones
        filtered: List[Tuple[Tuple[str, ...], float, float]] = []
        for seq, support, avg_f1 in candidates:
            is_subset = False
            for existing_seq, _, _ in filtered:
                if len(seq) < len(existing_seq) and _is_subsequence(seq, existing_seq):
                    is_subset = True
                    break
            if not is_subset:
                filtered.append((seq, support, avg_f1))

        # Create skills (up to max_skills)
        existing_seqs = {tuple(s.action_sequence) for s in self.skills}
        new_skills: List[InvestigationSkill] = []

        for seq, support, avg_f1 in filtered[:self.max_skills]:
            if seq in existing_seqs:
                # Update existing skill's support
                for s in self.skills:
                    if tuple(s.action_sequence) == seq:
                        s.support = support
                        s.avg_f1 = avg_f1
                continue

            name = _generate_skill_name(list(seq), len(self.skills) + len(new_skills))
            desc = _generate_skill_description(list(seq))

            skill = InvestigationSkill(
                name=name,
                action_sequence=list(seq),
                support=round(support, 4),
                avg_f1=round(avg_f1, 4),
                discovery_episode=current_episode,
                description=desc,
            )
            new_skills.append(skill)
            existing_seqs.add(seq)

        self.skills.extend(new_skills)

        # Prune to max_skills (keep highest support * avg_f1)
        if len(self.skills) > self.max_skills:
            self.skills.sort(key=lambda s: s.support * s.avg_f1, reverse=True)
            self.skills = self.skills[:self.max_skills]

        return new_skills

    # ------------------------------------------------------------------ #
    # Prompt injection                                                     #
    # ------------------------------------------------------------------ #

    def get_skill_prompt_injection(self) -> str:
        """
        Generate a prompt fragment that describes discovered skills.

        This is appended to the agent's system prompt so it can leverage
        learned investigation strategies.
        """
        if not self.skills:
            return ""

        lines = [
            "\nDISCOVERED INVESTIGATION SKILLS (proven effective patterns):"
        ]
        for i, skill in enumerate(self.skills, 1):
            seq_str = " → ".join(skill.action_sequence)
            lines.append(
                f"  SKILL {i}: {skill.name} "
                f"(success rate: {skill.support*100:.0f}%, avg F1: {skill.avg_f1:.2f})"
            )
            lines.append(f"    Pattern: {seq_str}")
            if skill.description:
                lines.append(f"    Strategy: {skill.description}")
        lines.append(
            "  When your investigation matches a skill pattern, follow it through."
        )
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # Persistence                                                          #
    # ------------------------------------------------------------------ #

    def save(self, path: Optional[str] = None) -> None:
        """Save skills library to JSON."""
        path = path or SKILLS_PATH
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        data = {
            "skills": [s.to_dict() for s in self.skills],
            "episode_count": self._episode_count,
            "total_episodes_logged": len(self._episode_log),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load(self, path: Optional[str] = None) -> None:
        """Load skills library from JSON."""
        path = path or SKILLS_PATH
        if not os.path.exists(path):
            return
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.skills = [InvestigationSkill.from_dict(s) for s in data.get("skills", [])]
        self._episode_count = data.get("episode_count", 0)

    # ------------------------------------------------------------------ #
    # Diagnostics                                                          #
    # ------------------------------------------------------------------ #

    def summary(self) -> str:
        """Human-readable summary of discovered skills."""
        if not self.skills:
            return "  No skills discovered yet."
        lines = [f"  Skill Library ({len(self.skills)} skills):"]
        for s in self.skills:
            lines.append(
                f"    {s.name:30s}  support={s.support:.2f}  "
                f"avg_f1={s.avg_f1:.2f}  len={len(s.action_sequence)}"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_subsequence(short: Tuple[str, ...], long: Tuple[str, ...]) -> bool:
    """Check if `short` is a contiguous subsequence of `long`."""
    s_len = len(short)
    for i in range(len(long) - s_len + 1):
        if long[i:i + s_len] == short:
            return True
    return False


def _generate_skill_name(actions: List[str], skill_id: int) -> str:
    """Generate a descriptive name for a skill based on its action composition."""
    action_set = set(actions)

    if "cross_reference_jurisdiction" in action_set and actions.count("trace_network") >= 2:
        return f"jurisdiction_sweep_{skill_id}"
    if actions.count("trace_network") >= 3:
        return f"deep_network_scan_{skill_id}"
    if actions[0] == "query_transactions" and "trace_network" in action_set:
        return f"triage_to_trace_{skill_id}"
    if "file_SAR" in action_set:
        return f"full_investigation_{skill_id}"
    if actions.count("query_transactions") >= 2:
        return f"multi_query_sweep_{skill_id}"
    if "cross_reference_jurisdiction" in action_set:
        return f"cross_border_check_{skill_id}"
    if "request_subpoena" in action_set:
        return f"subpoena_pipeline_{skill_id}"
    return f"investigation_pattern_{skill_id}"


def _generate_skill_description(actions: List[str]) -> str:
    """Generate a natural-language description of an investigation skill."""
    n_query = actions.count("query_transactions")
    n_trace = actions.count("trace_network")
    n_xref  = actions.count("cross_reference_jurisdiction")
    n_sar   = actions.count("file_SAR")
    n_sub   = actions.count("request_subpoena")
    n_look  = actions.count("lookup_entity")

    parts = []
    if n_query > 0:
        parts.append(f"Query {n_query} entities for transaction patterns")
    if n_trace > 0:
        parts.append(f"trace network {n_trace} hops deep")
    if n_look > 0:
        parts.append(f"lookup {n_look} entity profiles")
    if n_xref > 0:
        parts.append(f"cross-reference {n_xref} jurisdictions")
    if n_sub > 0:
        parts.append("obtain subpoena for restricted data")
    if n_sar > 0:
        parts.append("file SAR with accumulated evidence")

    if not parts:
        return "Investigation pattern"
    return ", then ".join(parts) + "."


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile, shutil

    print("=" * 70)
    print("HEIST Skill Discovery — Unit Test")
    print("=" * 70)

    # Temp dir for test output
    _tmp = tempfile.mkdtemp()
    test_skills_path = os.path.join(_tmp, "skills.json")

    lib = SkillLibrary(
        min_support=0.5,
        min_length=3,
        max_skills=5,
        mine_every=5,
        f1_success_threshold=0.5,
    )

    # ── Test 1: Record episodes ────────────────────────────────────────
    print("\n[1] Recording 15 synthetic episodes...")

    # Successful episodes with a common pattern
    common_pattern = ["query_transactions", "query_transactions", "trace_network",
                      "trace_network", "trace_network", "cross_reference_jurisdiction",
                      "cross_reference_jurisdiction", "file_SAR"]

    for i in range(10):
        # 10 successful episodes with variations of the common pattern
        actions = list(common_pattern)
        if i % 3 == 0:
            actions.insert(2, "lookup_entity")  # some variation
        lib.record_episode(actions, f1=0.7 + (i % 3) * 0.05, scheme_type="smurfing")

    for i in range(5):
        # 5 failed episodes with random actions
        actions = ["trace_network", "request_subpoena", "query_transactions",
                    "lookup_entity", "file_SAR"]
        lib.record_episode(actions, f1=0.2, scheme_type="layering")

    assert lib._episode_count == 15
    print(f"  ✓ Recorded {lib._episode_count} episodes")

    # ── Test 2: Mine skills ────────────────────────────────────────────
    print("\n[2] Mining skills...")
    new_skills = lib.mine_skills(current_episode=15)
    assert len(new_skills) > 0, "Should discover at least one skill"
    print(f"  ✓ Discovered {len(new_skills)} new skills:")
    for s in new_skills:
        seq_str = " → ".join(s.action_sequence)
        print(f"    {s.name}: {seq_str} (support={s.support:.2f}, F1={s.avg_f1:.2f})")

    # ── Test 3: Prompt injection ───────────────────────────────────────
    print("\n[3] Prompt injection...")
    prompt_fragment = lib.get_skill_prompt_injection()
    assert "DISCOVERED INVESTIGATION SKILLS" in prompt_fragment
    assert "SKILL 1" in prompt_fragment
    print(f"  ✓ Generated prompt fragment ({len(prompt_fragment)} chars)")
    print(f"  Preview: {prompt_fragment[:200]}...")

    # ── Test 4: JSON persistence ───────────────────────────────────────
    print("\n[4] JSON persistence...")
    lib.save(test_skills_path)
    assert os.path.exists(test_skills_path)

    lib2 = SkillLibrary()
    lib2.load(test_skills_path)
    assert len(lib2.skills) == len(lib.skills)
    assert lib2.skills[0].name == lib.skills[0].name
    print(f"  ✓ Round-trip: saved {len(lib.skills)} skills, loaded {len(lib2.skills)}")

    # ── Test 5: Summary ───────────────────────────────────────────────
    print("\n[5] Summary:")
    print(lib.summary())

    # Cleanup
    shutil.rmtree(_tmp, ignore_errors=True)

    print(f"\n{'='*70}")
    print("ALL SKILL DISCOVERY TESTS PASSED")
    print(f"{'='*70}")
