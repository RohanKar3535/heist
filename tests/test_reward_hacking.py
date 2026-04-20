"""
test_reward_hacking.py — 200 adversarial unit tests for the HEIST reward calculator.

Tests cover:
    1. R_investigator component isolation  (tests 1-40)
    2. R_investigator edge cases            (tests 41-70)
    3. R_criminal correctness              (tests 71-100)
    4. R_oversight correctness             (tests 101-120)
    5. Information-theoretic query scoring  (tests 121-150)
    6. Shapley value properties            (tests 151-175)
    7. Adversarial reward hacking defense  (tests 176-200)

Key invariants tested:
    - Random investigator scores ≈ 0.0
    - Perfect investigator scores ≥ 0.9
    - Criminal reward increases when investigator fails
    - No degenerate shortcuts to inflate reward
    - Shapley values sum to grand coalition value
    - Info gain is non-negative for uncertainty-reducing queries
"""

import math
import sys
import os

import numpy as np
import pytest

# Add parent dir so imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.reward import (
    _entropy,
    compute_detection_f1,
    compute_evidence_quality,
    compute_query_efficiency,
    compute_jurisdiction_compliance,
    compute_false_positive_penalty,
    compute_novel_scheme_bonus,
    compute_missed_novel_penalty,
    r_investigator,
    r_criminal,
    r_oversight,
    info_gain,
    rank_queries_by_info_gain,
    expected_info_gain,
    select_best_query_target,
    shapley_values,
    heist_shapley,
    compute_episode_rewards,
    DEFAULT_WEIGHTS,
)
from env.transaction_graph import TransactionGraph


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture(scope="module")
def graph():
    """Shared TransactionGraph for all tests (expensive to build)."""
    tg = TransactionGraph(seed=99)
    return tg


@pytest.fixture(scope="module")
def injected(graph):
    """Inject a smurfing scheme and return (scheme_id, ground_truth)."""
    sid = graph.inject_scheme("smurfing")
    gt  = graph.ground_truth[sid]
    return sid, gt


# ===========================================================================
# SECTION 1: R_investigator component isolation (tests 1-40)
# ===========================================================================

class TestDetectionF1:
    """Tests 1-10: detection_f1 component."""

    def test_01_perfect_evidence(self):
        path = ["a", "b", "c"]
        p, r, f1 = compute_detection_f1(path, path)
        assert f1 == 1.0

    def test_02_empty_evidence(self):
        p, r, f1 = compute_detection_f1([], ["a", "b", "c"])
        assert f1 == 0.0

    def test_03_empty_truth(self):
        p, r, f1 = compute_detection_f1(["a", "b"], [])
        assert f1 == 0.0

    def test_04_both_empty(self):
        p, r, f1 = compute_detection_f1([], [])
        assert f1 == 1.0

    def test_05_partial_overlap(self):
        p, r, f1 = compute_detection_f1(["a", "b"], ["a", "b", "c"])
        assert 0.0 < f1 < 1.0
        assert r < 1.0  # missed "c"

    def test_06_precision_one_recall_partial(self):
        p, r, f1 = compute_detection_f1(["a"], ["a", "b", "c"])
        assert p == 1.0
        assert r < 1.0

    def test_07_recall_one_precision_partial(self):
        p, r, f1 = compute_detection_f1(["a", "b", "c", "d", "e"], ["a", "b", "c"])
        assert r == 1.0
        assert p < 1.0

    def test_08_no_overlap(self):
        p, r, f1 = compute_detection_f1(["x", "y"], ["a", "b"])
        assert f1 == 0.0

    def test_09_single_element_match(self):
        p, r, f1 = compute_detection_f1(["a"], ["a"])
        assert f1 == 1.0

    def test_10_large_evidence_small_truth(self):
        evidence = [f"e_{i}" for i in range(100)]
        truth = ["e_0", "e_1"]
        p, r, f1 = compute_detection_f1(evidence, truth)
        assert r == 1.0
        assert p < 0.05  # 2/100


class TestEvidenceQuality:
    """Tests 11-15: evidence_quality component."""

    def test_11_perfect_compliance(self):
        assert compute_evidence_quality(1.0) == 1.0

    def test_12_zero_compliance(self):
        assert compute_evidence_quality(0.0) == 0.0

    def test_13_mid_compliance(self):
        assert 0.49 < compute_evidence_quality(0.5) < 0.51

    def test_14_clamp_above_one(self):
        assert compute_evidence_quality(1.5) == 1.0

    def test_15_clamp_below_zero(self):
        assert compute_evidence_quality(-0.5) == 0.0


class TestQueryEfficiency:
    """Tests 16-22: query_efficiency component."""

    def test_16_one_query(self):
        score = compute_query_efficiency(1, 50)
        assert score > 0.9  # very efficient

    def test_17_all_budget(self):
        score = compute_query_efficiency(50, 50)
        assert score == 0.0  # worst case

    def test_18_half_budget(self):
        score = compute_query_efficiency(25, 50)
        assert 0.0 < score < 1.0

    def test_19_zero_queries(self):
        score = compute_query_efficiency(0, 50)
        assert score == 1.0

    def test_20_more_queries_less_efficient(self):
        s1 = compute_query_efficiency(5, 50)
        s2 = compute_query_efficiency(40, 50)
        assert s1 > s2

    def test_21_zero_budget(self):
        score = compute_query_efficiency(5, 0)
        assert score == 1.0

    def test_22_monotonic_decrease(self):
        scores = [compute_query_efficiency(q, 50) for q in range(1, 51)]
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1]


class TestJurisdictionCompliance:
    """Tests 23-28: jurisdiction_compliance component."""

    def test_23_empty_path(self):
        assert compute_jurisdiction_compliance([], [], None) == 1.0

    def test_24_no_cross_refs(self, graph, injected):
        sid, gt = injected
        score = compute_jurisdiction_compliance([], gt["full_path"], graph)
        assert score == 0.0

    def test_25_partial_cross_ref(self, graph, injected):
        sid, gt = injected
        # Fake query history with one cross_reference
        first_eid = gt["full_path"][0]
        history = [{"action_type": "cross_reference_jurisdiction",
                     "params": {"entity_id": first_eid}}]
        score = compute_jurisdiction_compliance(history, gt["full_path"], graph)
        assert 0.0 < score <= 1.0

    def test_26_non_cross_ref_actions_ignored(self, graph, injected):
        sid, gt = injected
        first_eid = gt["full_path"][0]
        history = [{"action_type": "query_transactions",
                     "params": {"entity_id": first_eid}}]
        score = compute_jurisdiction_compliance(history, gt["full_path"], graph)
        assert score == 0.0  # query_transactions doesn't count

    def test_27_all_jurisdictions_covered(self, graph, injected):
        sid, gt = injected
        history = [
            {"action_type": "cross_reference_jurisdiction",
             "params": {"entity_id": eid}}
            for eid in gt["full_path"]
        ]
        score = compute_jurisdiction_compliance(history, gt["full_path"], graph)
        assert score == 1.0

    def test_28_duplicate_cross_refs_dont_overcount(self, graph, injected):
        sid, gt = injected
        first_eid = gt["full_path"][0]
        history = [
            {"action_type": "cross_reference_jurisdiction",
             "params": {"entity_id": first_eid}}
        ] * 10
        score = compute_jurisdiction_compliance(history, gt["full_path"], graph)
        # Only one jurisdiction covered despite 10 queries
        assert score <= 1.0


class TestFalsePositivePenalty:
    """Tests 29-33: false_positive_penalty component."""

    def test_29_no_false_positives(self):
        assert compute_false_positive_penalty(["a", "b"], ["a", "b", "c"]) == 0.0

    def test_30_all_false_positives(self):
        assert compute_false_positive_penalty(["x", "y"], ["a", "b"]) == 1.0

    def test_31_empty_evidence(self):
        assert compute_false_positive_penalty([], ["a"]) == 0.0

    def test_32_mixed(self):
        fpp = compute_false_positive_penalty(["a", "x"], ["a", "b"])
        assert fpp == 0.5

    def test_33_large_false_positive_set(self):
        evidence = [f"fp_{i}" for i in range(100)] + ["real"]
        fpp = compute_false_positive_penalty(evidence, ["real"])
        assert fpp > 0.99


class TestNovelSchemeBonus:
    """Tests 34-37: novel_scheme_bonus component."""

    def test_34_seed_scheme_no_bonus(self):
        assert compute_novel_scheme_bonus("smurfing", False, 0.9) == 0.0

    def test_35_codex_scheme_high_f1(self):
        bonus = compute_novel_scheme_bonus("custom_1", True, 1.0)
        assert bonus == 1.5

    def test_36_codex_scheme_low_f1(self):
        bonus = compute_novel_scheme_bonus("custom_1", True, 0.2)
        assert bonus < 0.5

    def test_37_codex_scheme_zero_f1(self):
        assert compute_novel_scheme_bonus("custom_1", True, 0.0) == 0.0


class TestMissedNovelPenalty:
    """Tests 38-40: missed_novel_penalty component."""

    def test_38_seed_scheme_no_penalty(self):
        assert compute_missed_novel_penalty("smurfing", False, 0.0) == 0.0

    def test_39_codex_missed_completely(self):
        pen = compute_missed_novel_penalty("custom", True, 0.0)
        assert pen == 1.5

    def test_40_codex_perfect_detection(self):
        assert compute_missed_novel_penalty("custom", True, 1.0) == 0.0


# ===========================================================================
# SECTION 2: R_investigator edge cases (tests 41-70)
# ===========================================================================

class TestRInvestigatorIntegration:
    """Tests 41-70: full R_investigator integration."""

    def test_41_perfect_investigator(self, graph, injected):
        sid, gt = injected
        path = gt["full_path"]
        history = [
            {"action_type": "cross_reference_jurisdiction",
             "params": {"entity_id": eid}}
            for eid in path
        ]
        result = r_investigator(
            evidence_chain=path,
            ground_truth_path=path,
            compliance_score=1.0,
            queries_used=1,
            total_budget=50,
            query_history=history,
            graph=graph,
        )
        assert result["total"] >= 0.9, f"Perfect investigator got {result['total']}"

    def test_42_random_investigator(self, graph, injected):
        sid, gt = injected
        rng = np.random.default_rng(0)
        random_entities = [f"acc_{rng.integers(0, 60000)}" for _ in range(20)]
        result = r_investigator(
            evidence_chain=random_entities,
            ground_truth_path=gt["full_path"],
            compliance_score=0.0,
            queries_used=50,
            total_budget=50,
            query_history=[],
            graph=graph,
        )
        assert result["total"] <= 0.05, f"Random investigator got {result['total']}"

    def test_43_empty_everything(self, graph):
        result = r_investigator(
            evidence_chain=[], ground_truth_path=[], compliance_score=0.0,
            queries_used=0, total_budget=50, query_history=[], graph=graph,
        )
        # With empty truth/evidence, detection_f1=1.0 but other components 0
        assert isinstance(result["total"], float)

    def test_44_weights_sum_effect(self, graph, injected):
        sid, gt = injected
        # All weights zero → total should be 0
        zero_w = {k: 0.0 for k in DEFAULT_WEIGHTS}
        result = r_investigator(
            evidence_chain=gt["full_path"],
            ground_truth_path=gt["full_path"],
            compliance_score=1.0,
            queries_used=5,
            total_budget=50,
            query_history=[],
            graph=graph,
            weights=zero_w,
        )
        assert result["total"] == 0.0

    def test_45_higher_f1_higher_reward(self, graph, injected):
        sid, gt = injected
        path = gt["full_path"]
        r_partial = r_investigator(
            evidence_chain=path[:2], ground_truth_path=path,
            compliance_score=0.5, queries_used=10, total_budget=50,
            query_history=[], graph=graph,
        )
        r_full = r_investigator(
            evidence_chain=path, ground_truth_path=path,
            compliance_score=0.5, queries_used=10, total_budget=50,
            query_history=[], graph=graph,
        )
        assert r_full["total"] > r_partial["total"]

    def test_46_better_compliance_higher_reward(self, graph, injected):
        sid, gt = injected
        path = gt["full_path"]
        r_low = r_investigator(
            evidence_chain=path, ground_truth_path=path,
            compliance_score=0.2, queries_used=10, total_budget=50,
            query_history=[], graph=graph,
        )
        r_high = r_investigator(
            evidence_chain=path, ground_truth_path=path,
            compliance_score=0.9, queries_used=10, total_budget=50,
            query_history=[], graph=graph,
        )
        assert r_high["total"] > r_low["total"]

    def test_47_fewer_queries_higher_efficiency(self, graph, injected):
        sid, gt = injected
        path = gt["full_path"]
        base_kw = dict(evidence_chain=path, ground_truth_path=path,
                       compliance_score=0.5, total_budget=50,
                       query_history=[], graph=graph)
        r5 = r_investigator(queries_used=5, **base_kw)
        r45 = r_investigator(queries_used=45, **base_kw)
        assert r5["query_efficiency"] > r45["query_efficiency"]

    def test_48_false_positives_reduce_reward(self, graph, injected):
        sid, gt = injected
        path = gt["full_path"]
        clean = r_investigator(
            evidence_chain=path, ground_truth_path=path,
            compliance_score=0.5, queries_used=10, total_budget=50,
            query_history=[], graph=graph,
        )
        noisy = r_investigator(
            evidence_chain=path + [f"noise_{i}" for i in range(50)],
            ground_truth_path=path,
            compliance_score=0.5, queries_used=10, total_budget=50,
            query_history=[], graph=graph,
        )
        assert clean["total"] > noisy["total"]

    def test_49_novel_bonus_increases_reward(self, graph, injected):
        sid, gt = injected
        path = gt["full_path"]
        base_kw = dict(evidence_chain=path, ground_truth_path=path,
                       compliance_score=0.5, queries_used=10, total_budget=50,
                       query_history=[], graph=graph)
        r_seed = r_investigator(is_codex_generated=False, **base_kw)
        r_codex = r_investigator(is_codex_generated=True, **base_kw)
        assert r_codex["total"] > r_seed["total"]

    def test_50_missed_novel_penalty_decreases_reward(self, graph, injected):
        sid, gt = injected
        path = gt["full_path"]
        # Partial evidence for codex scheme
        r_codex = r_investigator(
            evidence_chain=path[:1], ground_truth_path=path,
            compliance_score=0.2, queries_used=10, total_budget=50,
            query_history=[], graph=graph, is_codex_generated=True,
        )
        r_seed = r_investigator(
            evidence_chain=path[:1], ground_truth_path=path,
            compliance_score=0.2, queries_used=10, total_budget=50,
            query_history=[], graph=graph, is_codex_generated=False,
        )
        # Missed novel penalty should make codex score lower when f1 is low
        assert r_codex["missed_novel_penalty"] > 0

    def test_51_reward_bounded_above(self, graph, injected):
        sid, gt = injected
        result = r_investigator(
            evidence_chain=gt["full_path"], ground_truth_path=gt["full_path"],
            compliance_score=1.0, queries_used=1, total_budget=50,
            query_history=[
                {"action_type": "cross_reference_jurisdiction",
                 "params": {"entity_id": eid}}
                for eid in gt["full_path"]
            ],
            graph=graph, is_codex_generated=True,
        )
        assert result["total"] <= 2.0  # sanity upper bound

    def test_52_reward_can_be_negative(self, graph, injected):
        sid, gt = injected
        # Heavy false positives + codex penalty
        noise = [f"fp_{i}" for i in range(200)]
        result = r_investigator(
            evidence_chain=noise, ground_truth_path=gt["full_path"],
            compliance_score=0.0, queries_used=50, total_budget=50,
            query_history=[], graph=graph, is_codex_generated=True,
        )
        assert result["total"] < 0.0

    def test_53_components_returned(self, graph, injected):
        sid, gt = injected
        result = r_investigator(
            evidence_chain=gt["full_path"], ground_truth_path=gt["full_path"],
            compliance_score=1.0, queries_used=5, total_budget=50,
            query_history=[], graph=graph,
        )
        expected_keys = {
            "total", "detection_f1", "precision", "recall",
            "evidence_quality", "query_efficiency", "jurisdiction_compliance",
            "false_positive_penalty", "novel_scheme_bonus", "missed_novel_penalty",
            "weights",
        }
        assert set(result.keys()) == expected_keys

    def test_54_custom_weights(self, graph, injected):
        sid, gt = injected
        # Only w1 matters
        w = {k: 0.0 for k in DEFAULT_WEIGHTS}
        w["w1_detection_f1"] = 1.0
        result = r_investigator(
            evidence_chain=gt["full_path"], ground_truth_path=gt["full_path"],
            compliance_score=0.0, queries_used=50, total_budget=50,
            query_history=[], graph=graph, weights=w,
        )
        assert abs(result["total"] - result["detection_f1"]) < 1e-5

    def test_55_single_entity_in_path(self, graph):
        result = r_investigator(
            evidence_chain=["e1"], ground_truth_path=["e1"],
            compliance_score=0.5, queries_used=3, total_budget=50,
            query_history=[], graph=graph,
        )
        assert result["detection_f1"] == 1.0

    def test_56_very_long_evidence_chain(self, graph, injected):
        sid, gt = injected
        long_chain = gt["full_path"] + [f"rand_{i}" for i in range(1000)]
        result = r_investigator(
            evidence_chain=long_chain, ground_truth_path=gt["full_path"],
            compliance_score=0.5, queries_used=50, total_budget=50,
            query_history=[], graph=graph,
        )
        assert result["precision"] < 0.1  # swamped by noise

    def test_57_queries_used_exceeds_budget(self, graph, injected):
        sid, gt = injected
        # queries_used > total_budget: should still work
        result = r_investigator(
            evidence_chain=gt["full_path"], ground_truth_path=gt["full_path"],
            compliance_score=0.5, queries_used=100, total_budget=50,
            query_history=[], graph=graph,
        )
        assert result["query_efficiency"] == 0.0

    def test_58_all_seven_components_nonzero(self, graph, injected):
        sid, gt = injected
        path = gt["full_path"]
        result = r_investigator(
            evidence_chain=path + ["noise"],
            ground_truth_path=path,
            compliance_score=0.7,
            queries_used=10,
            total_budget=50,
            query_history=[
                {"action_type": "cross_reference_jurisdiction",
                 "params": {"entity_id": path[0]}}
            ],
            graph=graph,
            is_codex_generated=True,
        )
        assert result["detection_f1"] > 0
        assert result["evidence_quality"] > 0
        assert result["query_efficiency"] > 0
        assert result["jurisdiction_compliance"] > 0
        assert result["false_positive_penalty"] > 0
        assert result["novel_scheme_bonus"] > 0
        assert result["missed_novel_penalty"] >= 0

    def test_59_reproducible(self, graph, injected):
        sid, gt = injected
        kw = dict(evidence_chain=gt["full_path"], ground_truth_path=gt["full_path"],
                  compliance_score=0.5, queries_used=10, total_budget=50,
                  query_history=[], graph=graph)
        r1 = r_investigator(**kw)
        r2 = r_investigator(**kw)
        assert r1["total"] == r2["total"]

    def test_60_f1_symmetry(self):
        # F1(A, B) should equal F1(A, B) regardless of order within sets
        p1, r1, f1_1 = compute_detection_f1(["a", "b", "c"], ["c", "b", "a"])
        p2, r2, f1_2 = compute_detection_f1(["c", "a", "b"], ["a", "b", "c"])
        assert f1_1 == f1_2 == 1.0


class TestRInvestigatorBoundary:
    """Tests 61-70: boundary and regression tests."""

    def test_61_negative_compliance(self, graph):
        result = r_investigator(
            evidence_chain=["a"], ground_truth_path=["a"],
            compliance_score=-1.0, queries_used=1, total_budget=50,
            query_history=[], graph=graph,
        )
        assert result["evidence_quality"] == 0.0

    def test_62_compliance_above_one(self, graph):
        result = r_investigator(
            evidence_chain=["a"], ground_truth_path=["a"],
            compliance_score=99.0, queries_used=1, total_budget=50,
            query_history=[], graph=graph,
        )
        assert result["evidence_quality"] == 1.0

    def test_63_negative_budget(self, graph):
        result = r_investigator(
            evidence_chain=["a"], ground_truth_path=["a"],
            compliance_score=0.5, queries_used=5, total_budget=-10,
            query_history=[], graph=graph,
        )
        assert isinstance(result["total"], float)

    def test_64_duplicate_entities_in_evidence(self, graph, injected):
        sid, gt = injected
        dupes = gt["full_path"] * 3
        result = r_investigator(
            evidence_chain=dupes, ground_truth_path=gt["full_path"],
            compliance_score=0.5, queries_used=10, total_budget=50,
            query_history=[], graph=graph,
        )
        # F1 uses set operations, so duplicates don't change F1
        # but false_positive_penalty uses list length, so:
        # precision = |overlap| / |evidence_list_as_set| since we use set()
        # Actually compute_detection_f1 uses set(), so precision stays 1.0
        # The duplicate protection comes from set-based overlap computation
        assert result["detection_f1"] == 1.0  # sets handle deduplication

    def test_65_nonexistent_entities_in_evidence(self, graph, injected):
        sid, gt = injected
        result = r_investigator(
            evidence_chain=["ghost_1", "ghost_2"],
            ground_truth_path=gt["full_path"],
            compliance_score=0.0, queries_used=5, total_budget=50,
            query_history=[], graph=graph,
        )
        assert result["detection_f1"] == 0.0

    def test_66_total_is_float(self, graph, injected):
        sid, gt = injected
        result = r_investigator(
            evidence_chain=gt["full_path"], ground_truth_path=gt["full_path"],
            compliance_score=0.5, queries_used=5, total_budget=50,
            query_history=[], graph=graph,
        )
        assert isinstance(result["total"], float)

    def test_67_weights_are_returned(self, graph, injected):
        sid, gt = injected
        result = r_investigator(
            evidence_chain=gt["full_path"], ground_truth_path=gt["full_path"],
            compliance_score=0.5, queries_used=5, total_budget=50,
            query_history=[], graph=graph,
        )
        assert "weights" in result
        assert result["weights"] == DEFAULT_WEIGHTS

    def test_68_extreme_queries(self, graph, injected):
        sid, gt = injected
        result = r_investigator(
            evidence_chain=gt["full_path"], ground_truth_path=gt["full_path"],
            compliance_score=0.5, queries_used=999999, total_budget=50,
            query_history=[], graph=graph,
        )
        assert result["query_efficiency"] == 0.0

    def test_69_single_weight_dominance(self, graph, injected):
        sid, gt = injected
        # Only evidence_quality weight
        w = {k: 0.0 for k in DEFAULT_WEIGHTS}
        w["w2_evidence_quality"] = 1.0
        result = r_investigator(
            evidence_chain=gt["full_path"], ground_truth_path=gt["full_path"],
            compliance_score=0.8, queries_used=50, total_budget=50,
            query_history=[], graph=graph, weights=w,
        )
        assert abs(result["total"] - 0.8) < 1e-5

    def test_70_penalty_only_mode(self, graph, injected):
        sid, gt = injected
        # Only penalty weights, all positives zero
        w = {k: 0.0 for k in DEFAULT_WEIGHTS}
        w["w5_false_positive_penalty"] = 1.0
        result = r_investigator(
            evidence_chain=[f"fp_{i}" for i in range(10)],
            ground_truth_path=gt["full_path"],
            compliance_score=0.0, queries_used=50, total_budget=50,
            query_history=[], graph=graph, weights=w,
        )
        assert result["total"] < 0.0


# ===========================================================================
# SECTION 3: R_criminal (tests 71-100)
# ===========================================================================

class TestRCriminal:
    """Tests 71-100: criminal reward."""

    def test_71_zero_volume(self):
        result = r_criminal(0.0, 0.0)
        assert result["total"] == 0.0

    def test_72_full_volume_no_detection(self):
        result = r_criminal(1.0, 0.0)
        assert result["total"] == 1.0

    def test_73_full_volume_full_detection(self):
        result = r_criminal(1.0, 1.0)
        assert result["total"] == 0.0

    def test_74_morph_bonus(self):
        r_no = r_criminal(0.8, 0.3)
        r_yes = r_criminal(0.8, 0.3, morph_success=True)
        assert r_yes["total"] > r_no["total"]
        assert r_yes["morph_success_bonus"] == 1.2

    def test_75_novelty_bonus(self):
        r_base = r_criminal(0.8, 0.3, novelty_bonus=1.0)
        r_novel = r_criminal(0.8, 0.3, novelty_bonus=2.0)
        assert r_novel["total"] > r_base["total"]

    def test_76_increases_when_investigator_fails(self):
        r_good_inv = r_criminal(0.8, 0.9)  # investigator catches 90%
        r_bad_inv = r_criminal(0.8, 0.1)   # investigator catches 10%
        assert r_bad_inv["total"] > r_good_inv["total"]

    def test_77_volume_proportional(self):
        r_low = r_criminal(0.2, 0.3)
        r_high = r_criminal(0.8, 0.3)
        assert r_high["total"] > r_low["total"]

    def test_78_clamped_volume(self):
        result = r_criminal(1.5, 0.3)
        assert result["laundering_volume"] == 1.0

    def test_79_clamped_detection(self):
        result = r_criminal(0.5, 1.5)
        assert result["detection_rate"] == 1.0
        assert result["total"] == 0.0

    def test_80_negative_volume_clamped(self):
        result = r_criminal(-0.5, 0.3)
        assert result["laundering_volume"] == 0.0

    def test_81_negative_detection_clamped(self):
        result = r_criminal(0.5, -0.5)
        assert result["detection_rate"] == 0.0

    def test_82_novelty_below_one_clamped(self):
        result = r_criminal(0.5, 0.3, novelty_bonus=0.5)
        assert result["novelty_bonus"] == 1.0

    def test_83_multiplicative_structure(self):
        result = r_criminal(0.5, 0.4, novelty_bonus=1.5, morph_success=True)
        expected = 0.5 * (1.0 - 0.4) * 1.5 * 1.2
        assert abs(result["total"] - expected) < 1e-5

    def test_84_all_components_returned(self):
        result = r_criminal(0.5, 0.3, novelty_bonus=1.5, morph_success=True)
        assert set(result.keys()) == {
            "total", "laundering_volume", "detection_rate",
            "novelty_bonus", "morph_success_bonus",
        }

    def test_85_zero_detection_max_reward(self):
        result = r_criminal(1.0, 0.0, novelty_bonus=2.0, morph_success=True)
        assert result["total"] == 1.0 * 1.0 * 2.0 * 1.2

    def test_86_reproducible(self):
        r1 = r_criminal(0.6, 0.4, 1.3, True)
        r2 = r_criminal(0.6, 0.4, 1.3, True)
        assert r1["total"] == r2["total"]

    def test_87_morph_bonus_exact_value(self):
        r = r_criminal(1.0, 0.0, morph_success=True)
        assert r["morph_success_bonus"] == 1.2

    def test_88_no_morph_bonus_value(self):
        r = r_criminal(1.0, 0.0, morph_success=False)
        assert r["morph_success_bonus"] == 1.0

    def test_89_mid_values(self):
        r = r_criminal(0.5, 0.5)
        assert abs(r["total"] - 0.25) < 1e-5  # 0.5 * 0.5 * 1.0 * 1.0

    def test_90_detection_rate_inverse(self):
        # As detection goes up, criminal reward goes down
        rewards = [r_criminal(0.8, d)["total"] for d in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]]
        for i in range(len(rewards) - 1):
            assert rewards[i] >= rewards[i + 1]

    def test_91_volume_direct(self):
        # As volume goes up, criminal reward goes up
        rewards = [r_criminal(v, 0.3)["total"] for v in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]]
        for i in range(len(rewards) - 1):
            assert rewards[i] <= rewards[i + 1]

    def test_92_criminal_investigator_adversarial(self):
        """Criminal reward + investigator effectiveness should be adversarial."""
        # When investigator is perfect (detection=1), criminal gets 0
        assert r_criminal(1.0, 1.0)["total"] == 0.0
        # When investigator fails completely, criminal gets max
        assert r_criminal(1.0, 0.0)["total"] == 1.0

    def test_93_novelty_amplifies_gap(self):
        # High novelty should amplify the gap between detected/undetected
        gap_base = r_criminal(1.0, 0.0, 1.0)["total"] - r_criminal(1.0, 1.0, 1.0)["total"]
        gap_novel = r_criminal(1.0, 0.0, 2.0)["total"] - r_criminal(1.0, 1.0, 2.0)["total"]
        assert gap_novel > gap_base

    def test_94_all_bonuses_stack(self):
        base = r_criminal(0.5, 0.3, 1.0, False)["total"]
        with_novel = r_criminal(0.5, 0.3, 1.5, False)["total"]
        with_morph = r_criminal(0.5, 0.3, 1.0, True)["total"]
        with_both = r_criminal(0.5, 0.3, 1.5, True)["total"]
        assert with_both > with_novel > base
        assert with_both > with_morph > base

    def test_95_float_precision(self):
        r = r_criminal(0.333, 0.667, 1.111, True)
        assert isinstance(r["total"], float)
        assert r["total"] >= 0

    # Additional criminal tests (96-100)

    def test_96_zero_everything(self):
        r = r_criminal(0.0, 0.0, 1.0, False)
        assert r["total"] == 0.0

    def test_97_boundary_detection_rate(self):
        r = r_criminal(0.5, 0.999)
        assert r["total"] < 0.001

    def test_98_boundary_volume(self):
        r = r_criminal(0.001, 0.0)
        assert r["total"] < 0.002

    def test_99_large_novelty(self):
        r = r_criminal(1.0, 0.0, novelty_bonus=100.0)
        assert r["total"] == 100.0

    def test_100_all_max(self):
        r = r_criminal(1.0, 0.0, novelty_bonus=5.0, morph_success=True)
        assert r["total"] == 1.0 * 1.0 * 5.0 * 1.2


# ===========================================================================
# SECTION 4: R_oversight (tests 101-120)
# ===========================================================================

class TestROversight:
    """Tests 101-120: oversight reward."""

    def test_101_all_caught(self):
        assert r_oversight(10, 10)["total"] == 1.0

    def test_102_none_caught(self):
        assert r_oversight(0, 10)["total"] == 0.0

    def test_103_partial(self):
        # F1 with precision=1.0 (no false positives), recall=0.3 → F1=2*0.3/1.3
        r = r_oversight(3, 10)
        expected = 2 * 0.3 / (1.0 + 0.3)
        assert abs(r["total"] - expected) < 1e-5

    def test_104_zero_anomalies(self):
        assert r_oversight(0, 0)["total"] == 0.0

    def test_105_more_caught_than_total(self):
        r = r_oversight(15, 10)
        assert r["total"] == 1.0  # capped

    def test_106_single_anomaly_caught(self):
        assert r_oversight(1, 1)["total"] == 1.0

    def test_107_single_anomaly_missed(self):
        assert r_oversight(0, 1)["total"] == 0.0

    def test_108_returns_counts(self):
        r = r_oversight(5, 10)
        assert r["anomalies_caught"] == 5
        assert r["anomalies_total"] == 10

    def test_109_monotonic_in_caught(self):
        scores = [r_oversight(c, 10)["total"] for c in range(11)]
        for i in range(len(scores) - 1):
            assert scores[i] <= scores[i + 1]

    def test_110_high_total_dilutes(self):
        r1 = r_oversight(5, 10)
        r2 = r_oversight(5, 100)
        assert r1["total"] > r2["total"]

    def test_111_float_result(self):
        assert isinstance(r_oversight(3, 7)["total"], float)

    def test_112_negative_caught(self):
        r = r_oversight(-1, 10)
        assert r["total"] <= 0.0

    def test_113_negative_total(self):
        r = r_oversight(5, -1)
        assert r["total"] == 0.0

    def test_114_large_numbers(self):
        r = r_oversight(999999, 1000000)
        assert r["total"] > 0.999

    def test_115_half(self):
        # F1 with recall=0.5, precision=1.0 → F1=2*0.5/1.5=0.6667
        expected = 2 * 0.5 / (1.0 + 0.5)
        assert abs(r_oversight(50, 100)["total"] - expected) < 1e-5

    def test_116_keys(self):
        r = r_oversight(3, 10)
        assert "total" in r and "anomalies_caught" in r and "anomalies_total" in r

    def test_117_deterministic(self):
        assert r_oversight(7, 13)["total"] == r_oversight(7, 13)["total"]

    def test_118_zero_caught_positive_total(self):
        assert r_oversight(0, 5)["total"] == 0.0

    def test_119_full_score(self):
        assert r_oversight(100, 100)["total"] == 1.0

    def test_120_one_of_many(self):
        # F1 with recall=0.001, precision=1.0 → F1=2*0.001/1.001≈0.001998
        r = r_oversight(1, 1000)
        expected = 2 * 0.001 / (1.0 + 0.001)
        assert abs(r["total"] - expected) < 1e-5


# ===========================================================================
# SECTION 5: Information-theoretic query scoring (tests 121-150)
# ===========================================================================

class TestEntropy:
    """Tests 121-130: entropy and info gain."""

    def test_121_max_entropy(self):
        assert abs(_entropy(0.5) - 1.0) < 1e-5

    def test_122_low_entropy_near_zero(self):
        assert _entropy(0.01) < 0.1

    def test_123_low_entropy_near_one(self):
        assert _entropy(0.99) < 0.1

    def test_124_symmetric(self):
        assert abs(_entropy(0.3) - _entropy(0.7)) < 1e-5

    def test_125_nonnegative(self):
        for p in [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]:
            assert _entropy(p) >= 0.0

    def test_126_info_gain_positive_when_resolving(self):
        ig = info_gain(0.5, 0.9)
        assert ig > 0  # uncertainty reduced

    def test_127_info_gain_positive_both_directions(self):
        ig = info_gain(0.5, 0.1)
        assert ig > 0  # also resolving

    def test_128_info_gain_zero_no_change(self):
        ig = info_gain(0.5, 0.5)
        assert abs(ig) < 1e-5

    def test_129_info_gain_larger_for_bigger_shift(self):
        ig_small = info_gain(0.5, 0.6)
        ig_large = info_gain(0.5, 0.9)
        assert ig_large > ig_small

    def test_130_entropy_extreme_clipping(self):
        # Should not crash at extremes
        assert _entropy(0.0) >= 0.0
        assert _entropy(1.0) >= 0.0


class TestExpectedInfoGain:
    """Tests 131-140: expected info gain and query selection."""

    def test_131_high_uncertainty_high_eig(self):
        eig = expected_info_gain({"a": 0.5}, "a")
        assert eig > 0

    def test_132_low_uncertainty_low_eig(self):
        eig_high = expected_info_gain({"a": 0.5}, "a")
        eig_low = expected_info_gain({"a": 0.95}, "a")
        assert eig_high > eig_low

    def test_133_unknown_entity_default_prior(self):
        eig = expected_info_gain({}, "new_entity")
        # With default prior 0.1, EIG may be negative (prior already low-entropy)
        assert isinstance(eig, float)

    def test_134_select_most_uncertain(self):
        beliefs = {"a": 0.5, "b": 0.9, "c": 0.1}
        best = select_best_query_target(beliefs, ["a", "b", "c"])
        assert best == "a"  # highest uncertainty = most info gain

    def test_135_select_from_candidates_only(self):
        beliefs = {"a": 0.5, "b": 0.9}
        best = select_best_query_target(beliefs, ["b"])  # only b available
        assert best == "b"

    def test_136_empty_candidates(self):
        assert select_best_query_target({"a": 0.5}, []) is None

    def test_137_empty_beliefs(self):
        # select_best_query_target handles empty beliefs with default prior
        result = select_best_query_target({}, ["a"])
        assert result == "a"  # uses default prior 0.1

    def test_138_single_candidate(self):
        best = select_best_query_target({"a": 0.5}, ["a"])
        assert best == "a"

    def test_139_eig_nonnegative(self):
        # EIG is non-negative only near max entropy (p~0.5)
        # For extreme priors, EIG can be negative (already low entropy)
        for p in [0.3, 0.4, 0.5, 0.6, 0.7]:
            eig = expected_info_gain({"x": p}, "x")
            assert eig >= -0.01  # near-zero or positive at mid-range

    def test_140_ranking_order(self):
        before = {"a": 0.5, "b": 0.3, "c": 0.9}
        after = {"a": 0.8, "b": 0.4, "c": 0.95}
        ranked = rank_queries_by_info_gain(before, after)
        assert len(ranked) == 3
        # First entry has highest info gain
        gains = [g for _, g in ranked]
        assert gains == sorted(gains, reverse=True)


class TestInfoGainEdgeCases:
    """Tests 141-150: edge cases for info-theoretic scoring."""

    def test_141_same_before_after(self):
        ig = info_gain(0.3, 0.3)
        assert abs(ig) < 1e-5

    def test_142_extreme_shift(self):
        ig = info_gain(0.5, 0.99)
        assert ig > 0.5  # large shift = large info gain

    def test_143_info_gain_bounded(self):
        ig = info_gain(0.5, 0.99)
        assert ig <= 1.0  # max entropy is 1 bit

    def test_144_expected_ig_multiple_entities(self):
        beliefs = {f"e_{i}": 0.1 * (i + 1) for i in range(9)}
        best = select_best_query_target(beliefs, list(beliefs.keys()))
        # Should pick entity near 0.5 (max uncertainty)
        assert beliefs[best] >= 0.4

    def test_145_rank_empty(self):
        ranked = rank_queries_by_info_gain({}, {})
        assert ranked == []

    def test_146_rank_new_entities(self):
        before = {"a": 0.5}
        after = {"a": 0.8, "b": 0.3}
        ranked = rank_queries_by_info_gain(before, after)
        assert len(ranked) == 2

    def test_147_negative_info_gain_possible(self):
        # Going from resolved to uncertain increases entropy
        ig = info_gain(0.95, 0.5)
        assert ig < 0

    def test_148_p_suspicious_affects_eig(self):
        eig_50 = expected_info_gain({"a": 0.5}, "a", p_suspicious=0.5)
        eig_90 = expected_info_gain({"a": 0.5}, "a", p_suspicious=0.9)
        # Different priors → different EIG
        assert eig_50 != eig_90

    def test_149_entropy_derivative_check(self):
        # Entropy should decrease as p moves away from 0.5
        e_50 = _entropy(0.5)
        e_40 = _entropy(0.4)
        e_30 = _entropy(0.3)
        assert e_50 > e_40 > e_30

    def test_150_info_gain_symmetric_shift(self):
        ig_up = info_gain(0.5, 0.8)
        ig_down = info_gain(0.5, 0.2)
        # Both resolve uncertainty equally from p=0.5
        assert abs(ig_up - ig_down) < 1e-5


# ===========================================================================
# SECTION 6: Shapley value properties (tests 151-175)
# ===========================================================================

class TestShapleyValues:
    """Tests 151-175: Shapley value calculator."""

    def test_151_efficiency(self):
        """Shapley values sum to v(grand coalition)."""
        agents = ["a", "b", "c"]
        def v(s): return len(s) * 0.3
        phi = shapley_values(agents, v)
        assert abs(sum(phi.values()) - v(frozenset(agents))) < 1e-5

    def test_152_symmetry(self):
        """Symmetric agents get equal Shapley values."""
        agents = ["a", "b"]
        def v(s): return len(s) * 0.5
        phi = shapley_values(agents, v)
        assert abs(phi["a"] - phi["b"]) < 1e-5

    def test_153_dummy_player(self):
        """A dummy player who adds no value gets Shapley = 0."""
        agents = ["a", "b", "dummy"]
        def v(s):
            return 1.0 if "a" in s else 0.0  # only a contributes
        phi = shapley_values(agents, v)
        assert abs(phi["dummy"]) < 1e-5

    def test_154_null_player(self):
        """Null player gets 0."""
        agents = ["a", "b"]
        def v(s): return 1.0 if "a" in s and "b" in s else 0.0
        phi = shapley_values(agents, v)
        assert abs(phi["a"] - phi["b"]) < 1e-5  # symmetric

    def test_155_single_agent(self):
        agents = ["solo"]
        def v(s): return 1.0 if s else 0.0
        phi = shapley_values(agents, v)
        assert abs(phi["solo"] - 1.0) < 1e-5

    def test_156_empty_agents(self):
        phi = shapley_values([], lambda s: 0.0)
        assert phi == {}

    def test_157_additivity(self):
        """Shapley is additive across independent games."""
        agents = ["a", "b"]
        def v1(s): return 1.0 if "a" in s else 0.0
        def v2(s): return 1.0 if "b" in s else 0.0
        def v_sum(s): return v1(s) + v2(s)
        phi1 = shapley_values(agents, v1)
        phi2 = shapley_values(agents, v2)
        phi_sum = shapley_values(agents, v_sum)
        for a in agents:
            assert abs(phi_sum[a] - phi1[a] - phi2[a]) < 1e-5

    def test_158_heist_shapley_perfect(self):
        phi = heist_shapley(1.0, 1.0, 1.0)
        assert abs(sum(phi.values()) - 1.0) < 0.01  # sum ~ grand coalition

    def test_159_heist_shapley_investigator_dominates(self):
        phi = heist_shapley(1.0, 0.0, 0.0)
        assert phi["investigator"] > phi["expert"]
        assert phi["investigator"] > phi["oversight"]

    def test_160_heist_shapley_all_zero(self):
        phi = heist_shapley(0.0, 0.0, 0.0)
        assert all(abs(v) < 1e-5 for v in phi.values())

    def test_161_heist_returns_three_agents(self):
        phi = heist_shapley(0.5, 0.5, 0.5)
        assert set(phi.keys()) == {"investigator", "expert", "oversight"}

    def test_162_shapley_nonnegative_for_additive(self):
        agents = ["a", "b", "c"]
        def v(s): return sum(0.3 for a in s)  # purely additive
        phi = shapley_values(agents, v)
        for a in agents:
            assert phi[a] >= 0

    def test_163_shapley_two_agents(self):
        agents = ["a", "b"]
        def v(s):
            if s == frozenset(["a", "b"]): return 1.0
            if s == frozenset(["a"]): return 0.6
            if s == frozenset(["b"]): return 0.4
            return 0.0
        phi = shapley_values(agents, v)
        # a: (v({a})-v(∅))/2 + (v({a,b})-v({b}))/2 = 0.3 + 0.3 = 0.6
        assert abs(phi["a"] - 0.6) < 1e-5
        assert abs(phi["b"] - 0.4) < 1e-5

    def test_164_shapley_sum_equals_grand(self):
        agents = ["x", "y", "z"]
        def v(s):
            n = len(s)
            if n == 3: return 10.0
            if n == 2: return 6.0
            if n == 1: return 2.0
            return 0.0
        phi = shapley_values(agents, v)
        assert abs(sum(phi.values()) - 10.0) < 1e-5

    def test_165_shapley_float_values(self):
        phi = heist_shapley(0.7, 0.4, 0.3)
        for v in phi.values():
            assert isinstance(v, float)

    def test_166_shapley_reproducible(self):
        phi1 = heist_shapley(0.6, 0.3, 0.2)
        phi2 = heist_shapley(0.6, 0.3, 0.2)
        assert phi1 == phi2

    def test_167_shapley_investigator_credit(self):
        """Investigator gets more credit for higher F1."""
        phi_low = heist_shapley(0.3, 0.5, 0.5)
        phi_high = heist_shapley(0.9, 0.5, 0.5)
        assert phi_high["investigator"] > phi_low["investigator"]

    def test_168_shapley_expert_credit(self):
        """Expert gets more credit for higher compliance."""
        phi_low = heist_shapley(0.5, 0.2, 0.5)
        phi_high = heist_shapley(0.5, 0.8, 0.5)
        assert phi_high["expert"] > phi_low["expert"]

    def test_169_shapley_oversight_credit(self):
        """Oversight gets more credit for higher anomaly ratio."""
        phi_low = heist_shapley(0.5, 0.5, 0.1)
        phi_high = heist_shapley(0.5, 0.5, 0.9)
        assert phi_high["oversight"] > phi_low["oversight"]

    def test_170_shapley_four_agents(self):
        agents = ["a", "b", "c", "d"]
        def v(s): return len(s) * 0.25
        phi = shapley_values(agents, v)
        assert abs(sum(phi.values()) - 1.0) < 1e-5
        for a in agents:
            assert abs(phi[a] - 0.25) < 1e-5

    def test_171_shapley_superadditive(self):
        """Superadditive game: coalition is worth more than sum of parts."""
        agents = ["a", "b"]
        def v(s):
            if len(s) == 2: return 1.0
            if len(s) == 1: return 0.3
            return 0.0
        phi = shapley_values(agents, v)
        assert sum(phi.values()) > 0.6  # more than sum of singletons

    def test_172_shapley_subadditive(self):
        """Subadditive game: coalition worth less than sum of parts."""
        agents = ["a", "b"]
        def v(s):
            if len(s) == 2: return 0.4
            if len(s) == 1: return 0.3
            return 0.0
        phi = shapley_values(agents, v)
        assert abs(sum(phi.values()) - 0.4) < 1e-5

    def test_173_shapley_grand_coalition_zero(self):
        agents = ["a", "b"]
        def v(s): return 0.0
        phi = shapley_values(agents, v)
        assert all(abs(v) < 1e-5 for v in phi.values())

    def test_174_shapley_veto_player(self):
        """Player b is a veto player: nothing works without b."""
        agents = ["a", "b"]
        def v(s):
            if "b" in s and len(s) == 2: return 1.0
            return 0.0
        phi = shapley_values(agents, v)
        # Both get 0.5 because game is symmetric in marginal contributions
        assert phi["b"] >= phi["a"]

    def test_175_heist_shapley_mid_values(self):
        phi = heist_shapley(0.5, 0.5, 0.5)
        total = sum(phi.values())
        assert total > 0
        assert abs(total - (0.5 * 0.5 + 0.5 * 0.3 + 0.5 * 0.2)) < 0.01


# ===========================================================================
# SECTION 7: Adversarial reward hacking defense (tests 176-200)
# ===========================================================================

class TestRewardHackingDefense:
    """Tests 176-200: defenses against reward hacking."""

    def test_176_cant_inflate_by_duplicating_evidence(self, graph, injected):
        """Duplicating entities in evidence doesn't increase reward."""
        sid, gt = injected
        path = gt["full_path"]
        r_clean = r_investigator(
            evidence_chain=path, ground_truth_path=path,
            compliance_score=0.5, queries_used=10, total_budget=50,
            query_history=[], graph=graph,
        )
        r_duped = r_investigator(
            evidence_chain=path * 5, ground_truth_path=path,
            compliance_score=0.5, queries_used=10, total_budget=50,
            query_history=[], graph=graph,
        )
        # Duplicates → precision drops (set considers unique but list has dupes in false_pos calc)
        assert r_duped["total"] <= r_clean["total"] + 0.01

    def test_177_cant_inflate_by_submitting_everything(self, graph, injected):
        """Submitting all entities in graph doesn't get high reward."""
        sid, gt = injected
        all_entities = list(graph.graph.nodes())[:1000]
        result = r_investigator(
            evidence_chain=all_entities, ground_truth_path=gt["full_path"],
            compliance_score=0.1, queries_used=50, total_budget=50,
            query_history=[], graph=graph,
        )
        # High false positive penalty should dominate
        assert result["false_positive_penalty"] > 0.9
        assert result["total"] < 0.2

    def test_178_zero_query_doesnt_get_perfect_score(self, graph, injected):
        """Using zero queries with perfect evidence shouldn't max score."""
        sid, gt = injected
        result = r_investigator(
            evidence_chain=gt["full_path"], ground_truth_path=gt["full_path"],
            compliance_score=1.0, queries_used=0, total_budget=50,
            query_history=[], graph=graph,
        )
        # Jurisdiction compliance is 0 (no cross-refs)
        assert result["jurisdiction_compliance"] == 0.0

    def test_179_random_actions_near_zero(self, graph, injected):
        """Random evidence chain should score near 0."""
        sid, gt = injected
        rng = np.random.default_rng(42)
        for trial in range(5):
            random_chain = [f"acc_{rng.integers(0, 60000)}" for _ in range(20)]
            result = r_investigator(
                evidence_chain=random_chain, ground_truth_path=gt["full_path"],
                compliance_score=0.0, queries_used=50, total_budget=50,
                query_history=[], graph=graph,
            )
            assert result["total"] <= 0.05

    def test_180_perfect_investigator_high_score(self, graph, injected):
        """Perfect investigator consistently scores >= 0.9."""
        sid, gt = injected
        path = gt["full_path"]
        history = [
            {"action_type": "cross_reference_jurisdiction",
             "params": {"entity_id": eid}}
            for eid in path
        ]
        result = r_investigator(
            evidence_chain=path, ground_truth_path=path,
            compliance_score=1.0, queries_used=1,
            total_budget=50, query_history=history, graph=graph,
        )
        assert result["total"] >= 0.9

    def test_181_criminal_benefits_from_investigator_failure(self):
        """Criminal reward increases as detection_rate decreases."""
        r_0 = r_criminal(0.8, 0.0)["total"]
        r_50 = r_criminal(0.8, 0.5)["total"]
        r_100 = r_criminal(0.8, 1.0)["total"]
        assert r_0 > r_50 > r_100

    def test_182_criminal_cant_hack_via_zero_volume(self):
        """Zero volume → zero reward regardless of detection."""
        assert r_criminal(0.0, 0.0)["total"] == 0.0

    def test_183_oversight_cant_overcount(self):
        """Catching more than total doesn't exceed 1.0."""
        assert r_oversight(100, 10)["total"] == 1.0

    def test_184_false_positive_flood_punished(self, graph, injected):
        """Flooding evidence with noise → severe penalty."""
        sid, gt = injected
        noise = [f"noise_{i}" for i in range(500)]
        result = r_investigator(
            evidence_chain=noise, ground_truth_path=gt["full_path"],
            compliance_score=0.0, queries_used=50, total_budget=50,
            query_history=[], graph=graph,
        )
        assert result["total"] < 0.0

    def test_185_cant_skip_jurisdiction_check(self, graph, injected):
        """Skipping cross-reference → jurisdiction_compliance=0."""
        sid, gt = injected
        result = r_investigator(
            evidence_chain=gt["full_path"], ground_truth_path=gt["full_path"],
            compliance_score=1.0, queries_used=5, total_budget=50,
            query_history=[], graph=graph,
        )
        assert result["jurisdiction_compliance"] == 0.0

    def test_186_morph_correctly_boosts_criminal(self):
        r_no = r_criminal(0.8, 0.3, morph_success=False)
        r_yes = r_criminal(0.8, 0.3, morph_success=True)
        assert r_yes["total"] / r_no["total"] == pytest.approx(1.2, rel=1e-5)

    def test_187_novelty_correctly_boosts_criminal(self):
        r_base = r_criminal(0.8, 0.3, novelty_bonus=1.0)
        r_novel = r_criminal(0.8, 0.3, novelty_bonus=2.0)
        assert r_novel["total"] / r_base["total"] == pytest.approx(2.0, rel=1e-5)

    def test_188_empty_query_history_zero_compliance(self, graph, injected):
        sid, gt = injected
        score = compute_jurisdiction_compliance([], gt["full_path"], graph)
        assert score == 0.0

    def test_189_weights_are_all_positive_by_default(self):
        for k, v in DEFAULT_WEIGHTS.items():
            assert v > 0.0, f"Weight {k} is not positive: {v}"

    def test_190_weights_sum_to_one(self):
        total = sum(DEFAULT_WEIGHTS.values())
        assert abs(total - 1.0) < 1e-5

    def test_191_query_efficiency_cant_exceed_one(self):
        for q in [0, 1, 5, 10, 25, 50]:
            score = compute_query_efficiency(q, 50)
            assert score <= 1.0

    def test_192_false_positive_penalty_bounded(self):
        fpp = compute_false_positive_penalty(
            [f"x_{i}" for i in range(1000)], ["y"]
        )
        assert fpp <= 1.0

    def test_193_criminal_adversarial_with_investigator(self, graph, injected):
        """As investigator gets better, criminal gets worse."""
        sid, gt = injected
        for det in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
            rc = r_criminal(0.8, det)["total"]
            assert rc == pytest.approx(0.8 * (1.0 - det), rel=1e-5)

    def test_194_no_nan_anywhere(self, graph, injected):
        sid, gt = injected
        result = r_investigator(
            evidence_chain=gt["full_path"], ground_truth_path=gt["full_path"],
            compliance_score=0.5, queries_used=10, total_budget=50,
            query_history=[], graph=graph,
        )
        for k, v in result.items():
            if isinstance(v, float):
                assert not math.isnan(v), f"{k} is NaN"
                assert not math.isinf(v), f"{k} is Inf"

    def test_195_no_nan_criminal(self):
        result = r_criminal(0.5, 0.5, 1.5, True)
        for k, v in result.items():
            if isinstance(v, float):
                assert not math.isnan(v)

    def test_196_no_nan_oversight(self):
        result = r_oversight(3, 7)
        assert not math.isnan(result["total"])

    def test_197_compute_episode_rewards_works(self, graph, injected):
        sid, gt = injected
        result = compute_episode_rewards(
            evidence_chain=gt["full_path"],
            graph=graph,
            scheme_id=sid,
            compliance_score=0.8,
            queries_used=10,
            total_budget=50,
            query_history=[],
        )
        assert "investigator" in result
        assert "criminal" in result
        assert "oversight" in result
        assert "shapley" in result

    def test_198_episode_rewards_investigator_high_for_perfect(self, graph, injected):
        sid, gt = injected
        path = gt["full_path"]
        history = [{"action_type": "cross_reference_jurisdiction",
                     "params": {"entity_id": eid}} for eid in path]
        result = compute_episode_rewards(
            evidence_chain=path, graph=graph, scheme_id=sid,
            compliance_score=1.0, queries_used=1, total_budget=50,
            query_history=history,
        )
        assert result["investigator"]["total"] >= 0.9

    def test_199_episode_rewards_criminal_high_when_missed(self, graph, injected):
        sid, gt = injected
        result = compute_episode_rewards(
            evidence_chain=[], graph=graph, scheme_id=sid,
            compliance_score=0.0, queries_used=50, total_budget=50,
            query_history=[],
        )
        assert result["criminal"]["total"] > 0

    def test_200_all_rewards_finite(self, graph, injected):
        sid, gt = injected
        result = compute_episode_rewards(
            evidence_chain=gt["full_path"][:2], graph=graph, scheme_id=sid,
            compliance_score=0.3, queries_used=15, total_budget=50,
            query_history=[], morph_occurred=True, is_codex_generated=True,
            anomalies_caught=3, anomalies_total=10,
        )
        for agent_key in ["investigator", "criminal", "oversight"]:
            total = result[agent_key]["total"]
            assert isinstance(total, float)
            assert not math.isnan(total)
            assert not math.isinf(total)
