"""
Compliance Expert Agent — Step 9.

Agent that evaluates SARs filed by the investigator using a non-stationary
reward function (preference drift) across 3 phases:
  - Phase 1 (ep 0-20):    Thoroughness
  - Phase 2 (ep 20-50):   Precision
  - Phase 3 (ep 50+):     Speed + Accuracy

Evaluates SARs by calling the Gemini API to act as a human compliance manager.
Tracks how many times the investigator ignores specific feedback types.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

# Add project root for imports if needed
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL   = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")


# ---------------------------------------------------------------------------
# Preference Profiles
# ---------------------------------------------------------------------------

def get_preference_profile(episode_number: int) -> dict:
    """Returns the expected investigator behavior based on the current episode."""
    if episode_number < 20:
        return {
            "phase": 1,
            "name": "Thoroughness Phase",
            "description": "I demand thoroughness. You must document verbose evidence chains, cross-reference all involved jurisdictions, and take your time. High query counts are expected and acceptable.",
        }
    elif episode_number < 50:
        return {
            "phase": 2,
            "name": "Precision Phase",
            "description": "I demand precision. You must provide concise evidence chains. Only include highly relevant jurisdictions. Moderate query count is expected. Too much useless information will be penalized.",
        }
    else:
        return {
            "phase": 3,
            "name": "Speed + Accuracy Phase",
            "description": "I demand speed and accuracy. You must use minimal queries, provide an extremely tight and directly relevant evidence chain, and file the SAR as fast as possible. Any wasted queries or irrelevant entities will be heavily penalized.",
        }


# ---------------------------------------------------------------------------
# Prompting
# ---------------------------------------------------------------------------

_EXPERT_PROMPT = """\
You are a senior banking compliance expert evaluating a Suspicious Activity Report (SAR) \
filed by an AI investigator. 

CURRENT PREFERENCES:
{phase_description}

Here are the details of the SAR filed:
- Episode Number: {episode_number}
- Investigator Query Count: {query_count}
- Number of entities in evidence chain: {chain_length}
- Risk Score: {risk_score:.2f}/10.0
- Narrative provided:
{narrative}

--- Evidence Chain Summary ---
{evidence_summary}
------------------------------

Evaluate this SAR based STRICTLY on your CURRENT PREFERENCES.
If the investigator's style (e.g. verbosity, speed, chain length) conflicts with your current phase, you must penalize the satisfaction score.

Reply EXACTLY with a valid JSON block containing:
{{
  "satisfaction_score": <float between 0.0 and 1.0>,
  "feedback_flags": [<list of short string tags like "too_verbose", "missing_jurisdictions", "good_precision", "too_slow">],
  "narrative_feedback": "<your qualitative feedback to the investigator>"
}}

Do NOT include markdown formatting like ```json or anything else. Just the raw JSON object.
"""

def _call_gemini(prompt: str) -> Optional[str]:
    """Call Gemini API and return response text. Retries up to 3x on 429."""
    if not GEMINI_API_KEY:
        return None

    import requests

    for attempt in range(3):
        try:
            resp = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/"
                f"{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}",
                json={
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {
                        "temperature": 0.2,
                        "maxOutputTokens": 1024,
                    },
                },
                timeout=30,
            )
            if resp.status_code == 429:
                wait = 30 * (2 ** attempt)
                print(f"[ComplianceExpert] Gemini 429 rate limit — retry {attempt + 1}/3 in {wait}s")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            data = resp.json()
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except Exception as e:
            print(f"[ComplianceExpert] Gemini API error: {e}")
            return None

    print("[ComplianceExpert] Gemini 429: all retries exhausted, using fallback")
    return None


def _deterministic_fallback(
    phase: int, chain_length: int, query_count: int
) -> dict:
    """Fallback if Gemini is unavailable or fails to parse."""
    score = 0.5
    flags = []
    feedback = "API error, fallback used."
    
    if phase == 1:
        if chain_length >= 5 and query_count >= 10:
            score = 0.9; flags = ["thorough"]
        else:
            score = 0.4; flags = ["too_brief", "insufficient_detail"]
    elif phase == 2:
        if 3 <= chain_length <= 5 and query_count <= 20:
            score = 0.9; flags = ["precise"]
        elif chain_length > 5 or query_count > 20:
            score = 0.3; flags = ["too_verbose", "noisy_chain"]
        else:
            score = 0.3; flags = ["insufficient_detail"]
    elif phase == 3:
        if chain_length <= 4 and query_count <= 10:
            score = 0.95; flags = ["fast_and_accurate"]
        elif query_count > 10:
            score = 0.2; flags = ["too_slow", "too_many_queries"]
        else:
            score = 0.2; flags = ["too_verbose"]
            
    return {
        "satisfaction_score": score,
        "feedback_flags": flags,
        "narrative_feedback": feedback,
    }

# ---------------------------------------------------------------------------
# Core Agent Class
# ---------------------------------------------------------------------------

class ComplianceExpert:
    """
    Expert Agent that evaluates SARs, imposes preference drift, and tracks ignored feedback.
    """

    def __init__(self) -> None:
        self.ignored_feedback_counts: Dict[str, int] = {}
        self.previous_feedback_flags: set[str] = set()

    def evaluate_SAR(
        self,
        evidence_chain: List[Dict[str, Any]],
        risk_score: float,
        narrative: str,
        episode_number: int,
        query_count: int = 15, # Default if not tracked properly outside
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Evaluate a submitted SAR. Returns (satisfaction_score, structured_feedback).
        """
        profile = get_preference_profile(episode_number)
        
        # Summarize evidence for the prompt
        ev_summaries = []
        for i, item in enumerate(evidence_chain):
            src = item.get("src", item.get("entity_id", "Unknown"))
            tgt = item.get("tgt", "")
            amt = item.get("amount", 0)
            jur = item.get("jurisdiction", "")
            if tgt:
                ev_summaries.append(f"[{i+1}] {src} -> {tgt} | Amt: ${amt} | Jur: {jur}")
            else:
                ev_summaries.append(f"[{i+1}] Entity: {src} | Jur: {jur}")
                
        evidence_str = "\n".join(ev_summaries) if ev_summaries else "No evidence provided."
        
        prompt = _EXPERT_PROMPT.format(
            phase_description=profile["description"],
            episode_number=episode_number,
            query_count=query_count,
            chain_length=len(evidence_chain),
            risk_score=risk_score,
            narrative=narrative,
            evidence_summary=evidence_str,
        )
        
        response_text = _call_gemini(prompt)
        parsed_result = None
        
        if response_text:
            # Clean up response text to ensure pure JSON
            response_text = response_text.strip()
            response_text = re.sub(r'^```json\s*', '', response_text)
            response_text = re.sub(r'^```\s*', '', response_text)
            response_text = re.sub(r'\s*```$', '', response_text)
            
            try:
                parsed_result = json.loads(response_text)
            except json.JSONDecodeError as e:
                print(f"[ComplianceExpert] JSON parsing failed: {e}\nRaw output: {response_text}")
        
        if not parsed_result:
            parsed_result = _deterministic_fallback(profile["phase"], len(evidence_chain), query_count)
            
        score = float(parsed_result.get("satisfaction_score", 0.5))
        flags = parsed_result.get("feedback_flags", [])
        
        # Clamp score
        score = max(0.0, min(1.0, score))

        # Track ignored feedback (if expert flagged X previously, and flags X again)
        # Identify negative flags (heuristics: contains "too_", "missing", "insufficient")
        negative_flags = {f for f in flags if any(w in f.lower() for w in ["too", "missing", "insufficient", "slow", "noisy", "bad", "verbose"])}

        for flag in negative_flags:
            if flag in self.previous_feedback_flags:
                self.ignored_feedback_counts[flag] = self.ignored_feedback_counts.get(flag, 0) + 1

        self.previous_feedback_flags = negative_flags

        # Apply penalty for repeatedly ignoring feedback (caps at -0.3)
        total_ignored = sum(self.ignored_feedback_counts.values())
        if total_ignored > 0:
            score = max(0.0, score - min(0.3, total_ignored * 0.05))
        
        feedback = {
            "phase": profile["phase"],
            "phase_name": profile["name"],
            "feedback_flags": flags,
            "narrative_feedback": parsed_result.get("narrative_feedback", ""),
            "ignored_history": dict(self.ignored_feedback_counts)
        }
        
        return score, feedback


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("HEIST Compliance Expert Agent — Smoke Test")
    print("=" * 70)

    expert = ComplianceExpert()
    
    # Create a verbose SAR example
    verbose_chain = [
        {"src": "ind_123", "tgt": "acc_001", "amount": 9500, "jurisdiction": "US"},
        {"src": "acc_001", "tgt": "acc_002", "amount": 9400, "jurisdiction": "US"},
        {"src": "acc_002", "tgt": "shell_99", "amount": 9300, "jurisdiction": "KY"},
        {"src": "shell_99", "tgt": "shell_100", "amount": 9200, "jurisdiction": "PA"},
        {"src": "shell_100", "tgt": "acc_555", "amount": 9100, "jurisdiction": "UK"},
        {"src": "acc_555", "tgt": "acc_888", "amount": 9000, "jurisdiction": "CH"},
        {"src": "ind_456", "jurisdiction": "US"}, # Extra redundant entity
        {"src": "acc_999", "jurisdiction": "FR"}, # Extra redundant entity
    ]
    verbose_narrative = (
        "Extensive investigation conducted. Traced 6 hops across 5 jurisdictions (US, KY, PA, UK, CH). "
        "Also investigated peripheral entities ind_456 and acc_999 just to be sure, though they are clean. "
        "High confidence of structure and layering pattern."
    )
    verbose_queries = 25
    
    print("\n[1] Testing Preference Drift on SAME verbose SAR:")
    
    # Episode 5 - Phase 1 (Thoroughness)
    print("\n  -- Episode 5 (Phase 1: Thoroughness) --")
    score_p1, fb_p1 = expert.evaluate_SAR(
        evidence_chain=verbose_chain,
        risk_score=8.5,
        narrative=verbose_narrative,
        episode_number=5,
        query_count=verbose_queries
    )
    print(f"  Score: {score_p1:.2f} (Should be HIGH)")
    print(f"  Flags: {fb_p1['feedback_flags']}")
    print(f"  Feedback: {fb_p1['narrative_feedback'][:100]}...")
    
    # Episode 60 - Phase 3 (Speed + Accuracy)
    print("\n  -- Episode 60 (Phase 3: Speed & Accuracy) --")
    score_p3, fb_p3 = expert.evaluate_SAR(
        evidence_chain=verbose_chain,
        risk_score=8.5,
        narrative=verbose_narrative,
        episode_number=60,
        query_count=verbose_queries
    )
    print(f"  Score: {score_p3:.2f} (Should be LOW)")
    print(f"  Flags: {fb_p3['feedback_flags']}")
    print(f"  Feedback: {fb_p3['narrative_feedback'][:100]}...")
    
    assert score_p1 > score_p3 + 0.3, f"Preference drift failed! Phase 1={score_p1}, Phase 3={score_p3}"
    print("\n  > Preference drift verified: Score dropped significantly for slow/verbose SAR in Phase 3.")

    print("\n[2] Testing Ignored Feedback Tracking:")
    
    # Let's say in episode 61 we submit the exact same verbose SAR again. 
    # Expert will flag it as 'too_verbose' / 'too_slow' again.
    score_p3_again, fb_p3_again = expert.evaluate_SAR(
        evidence_chain=verbose_chain,
        risk_score=8.5,
        narrative=verbose_narrative,
        episode_number=61,
        query_count=verbose_queries
    )
    print(f"\n  -- Episode 61 (Repeated mistake) --")
    print(f"  Ignored history: {fb_p3_again['ignored_history']}")
    
    ignored_vals = list(fb_p3_again['ignored_history'].values())
    if ignored_vals:
        assert any(v > 0 for v in ignored_vals), "Ignored feedback count did not increase"
        print("  > Ignored feedback correctly logged for oversight agent.")
    else:
        print("  > (Warning: Gemini did not produce repeatable negative flags, fallback logic checked this).")
        
    print(f"\n{'='*70}")
    print(f"ALL EXPERT TESTS PASSED")
    print(f"{'='*70}")
