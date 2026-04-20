"""
Scenario Library — Step 6.

30 hand-crafted scenarios across 4 difficulty tiers + 4 Codex placeholders.

Each scenario defines:
    - scheme_id, scheme_type, difficulty, tier
    - Injection params for TransactionGraph.inject_scheme()
    - Ground truth metadata for grading
    - correct_SAR_elements, ground_truth_causal_chain
    - jurisdictions_used (set of required jurisdictions to cross-ref)

Tiers:
    Tier 1 — Easy   (8):  single-scheme, 2-3 hops, 1-2 jurisdictions
    Tier 2 — Medium (10): 3-5 hops, multiple jurisdictions, some misdirection
    Tier 3 — Hard   (8):  6-8 hops, multi-scheme, shared nodes, circular paths
    Tier 4 — Extreme (4): reserved for Criminal Codex generated schemes
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

try:
    from .transaction_graph import TransactionGraph, SCHEME_TYPES
except ImportError:
    from transaction_graph import TransactionGraph, SCHEME_TYPES


# ---------------------------------------------------------------------------
# Scenario definition
# ---------------------------------------------------------------------------

@dataclass
class Scenario:
    """Single scenario in the library."""

    # Identity
    scenario_id: str
    name: str
    description: str

    # Injection spec — one or more scheme injections
    # Each entry: (scheme_type, params_dict)
    injections: List[Tuple[str, Dict[str, Any]]]

    # Metadata
    tier: int                        # 1=Easy, 2=Medium, 3=Hard, 4=Extreme
    difficulty: str                  # "easy", "medium", "hard", "extreme"
    expected_hops: int               # total hops across all injections
    jurisdictions_used: List[str]    # jurisdictions the investigator should cross-ref
    num_schemes: int = 1             # how many simultaneous schemes

    # Grading
    correct_SAR_elements: List[str] = field(default_factory=list)
    ground_truth_causal_chain: str = ""

    # Codex flag (for Tier 4)
    is_codex_generated: bool = False
    codex_code: Optional[str] = None

    def inject(self, graph: TransactionGraph) -> List[Dict[str, Any]]:
        """
        Inject all schemes for this scenario into the graph.
        Returns list of ground_truth dicts (one per injection).
        """
        results: List[Dict[str, Any]] = []
        for scheme_type, params in self.injections:
            sid = graph.inject_scheme(scheme_type, params)
            gt = graph.ground_truth[sid]
            gt["scenario_id"] = self.scenario_id
            gt["difficulty"] = self.difficulty
            gt["tier"] = self.tier
            gt["jurisdictions_used"] = self.jurisdictions_used
            gt["correct_SAR_elements"] = self.correct_SAR_elements
            gt["ground_truth_causal_chain"] = self.ground_truth_causal_chain
            gt["is_codex_generated"] = self.is_codex_generated
            results.append(gt)
        return results


# ---------------------------------------------------------------------------
# Tier 1 — Easy (8 scenarios)
# ---------------------------------------------------------------------------

TIER_1_EASY: List[Scenario] = [
    Scenario(
        scenario_id="T1_01_structuring",
        name="Basic Structuring / Smurfing",
        description=(
            "Individual splits $50k into 6 deposits just below $10k reporting "
            "threshold across smurf accounts, then aggregates to a single account."
        ),
        injections=[("smurfing", {"n_smurfs": 6})],
        tier=1, difficulty="easy", expected_hops=2,
        jurisdictions_used=["US"],
        correct_SAR_elements=[
            "structured_deposits", "below_CTR_threshold", "aggregation_pattern",
        ],
        ground_truth_causal_chain=(
            "Individual makes 6 cash deposits of $9k-$9.9k each → "
            "Smurf accounts aggregate to single destination → "
            "Structuring to avoid CTR filing."
        ),
    ),
    Scenario(
        scenario_id="T1_02_shell_passthrough",
        name="Single Shell Pass-Through",
        description=(
            "Individual moves funds through 2 offshore shell companies to "
            "a clean account. Classic placement → layering → integration."
        ),
        injections=[("shell_company", {"n_shells": 2})],
        tier=1, difficulty="easy", expected_hops=3,
        jurisdictions_used=["KY", "PA"],
        correct_SAR_elements=[
            "shell_company_usage", "offshore_jurisdiction", "pass_through",
        ],
        ground_truth_causal_chain=(
            "Individual deposits illicit funds → Shell company A (Cayman Islands) → "
            "Shell company B (Panama) → Clean integration account."
        ),
    ),
    Scenario(
        scenario_id="T1_03_round_trip",
        name="Round-Trip Transaction",
        description=(
            "Account sends funds through 3 layering hops and back to a "
            "related account, disguising the circular flow."
        ),
        injections=[("layering", {"n_hops": 3})],
        tier=1, difficulty="easy", expected_hops=4,
        jurisdictions_used=["US", "UK"],
        correct_SAR_elements=[
            "circular_flow", "layering_pattern", "related_accounts",
        ],
        ground_truth_causal_chain=(
            "Source account → 3 shell intermediaries → Sink account. "
            "Circular pattern disguises origin of funds."
        ),
    ),
    Scenario(
        scenario_id="T1_04_cash_front",
        name="Cash-Intensive Front Business",
        description=(
            "Small smurfing scheme using 3 smurf accounts to aggregate "
            "cash deposits from a front business."
        ),
        injections=[("smurfing", {"n_smurfs": 3})],
        tier=1, difficulty="easy", expected_hops=2,
        jurisdictions_used=["US"],
        correct_SAR_elements=[
            "cash_intensive", "structured_deposits", "front_business",
        ],
        ground_truth_causal_chain=(
            "Individual (front business operator) → 3 smurf accounts → "
            "Aggregation account. Cash-intensive business provides cover story."
        ),
    ),
    Scenario(
        scenario_id="T1_05_two_hop_layering",
        name="2-Hop Layering",
        description=(
            "Simple 2-hop layering through shell companies. "
            "Minimal obfuscation."
        ),
        injections=[("layering", {"n_hops": 2})],
        tier=1, difficulty="easy", expected_hops=3,
        jurisdictions_used=["CH", "LU"],
        correct_SAR_elements=[
            "layering_pattern", "shell_intermediary", "cross_border",
        ],
        ground_truth_causal_chain=(
            "Source account → Shell company A → Shell company B → Sink account. "
            "Classic 2-hop layering with minimal obfuscation."
        ),
    ),
    Scenario(
        scenario_id="T1_06_invoice_fraud",
        name="Invoice Fraud",
        description=(
            "Trade-based money laundering through 2 intermediaries. "
            "Over-invoiced trade finance transactions."
        ),
        injections=[("trade_based", {"n_intermediaries": 2})],
        tier=1, difficulty="easy", expected_hops=3,
        jurisdictions_used=["HK", "SG"],
        correct_SAR_elements=[
            "trade_finance", "over_invoicing", "mismatched_values",
        ],
        ground_truth_causal_chain=(
            "Shell company → 2 intermediaries (over-invoiced goods) → "
            "Sink account. Invoice values deviate ±30% from fair market."
        ),
    ),
    Scenario(
        scenario_id="T1_07_crypto_deposit",
        name="Crypto Exchange Deposit",
        description=(
            "Funds moved through 2 crypto exchanges to obscure trail. "
            "Simple crypto-based placement."
        ),
        injections=[("crypto_mixing", {"n_hops": 2})],
        tier=1, difficulty="easy", expected_hops=3,
        jurisdictions_used=["MT", "SG"],
        correct_SAR_elements=[
            "crypto_transfer", "exchange_hopping", "jurisdiction_shopping",
        ],
        ground_truth_causal_chain=(
            "Source account → Crypto exchange A → Crypto exchange B → "
            "Clean account. Multiple exchange hops obscure origin."
        ),
    ),
    Scenario(
        scenario_id="T1_08_trade_laundering",
        name="Basic Trade-Based Laundering",
        description=(
            "Trade-based laundering with 3 intermediaries. "
            "Classic trade mispricing scheme."
        ),
        injections=[("trade_based", {"n_intermediaries": 3})],
        tier=1, difficulty="easy", expected_hops=4,
        jurisdictions_used=["HK", "SG", "US"],
        correct_SAR_elements=[
            "trade_finance", "mispricing", "multi_intermediary",
        ],
        ground_truth_causal_chain=(
            "Shell company → 3 intermediaries (mispriced trade invoices) → "
            "Sink account. Consistent ±30% invoice variance across hops."
        ),
    ),
]


# ---------------------------------------------------------------------------
# Tier 2 — Medium (10 scenarios)
# ---------------------------------------------------------------------------

TIER_2_MEDIUM: List[Scenario] = [
    Scenario(
        scenario_id="T2_01_three_hop_shell",
        name="3-Hop Shell Chain",
        description=(
            "3-layer shell company chain through offshore jurisdictions. "
            "Requires tracing through multiple corporate structures."
        ),
        injections=[("shell_company", {"n_shells": 3})],
        tier=2, difficulty="medium", expected_hops=4,
        jurisdictions_used=["KY", "VG", "CH"],
        correct_SAR_elements=[
            "shell_chain", "offshore_network", "corporate_layering",
        ],
        ground_truth_causal_chain=(
            "Individual → Shell A (Cayman) → Shell B (BVI) → "
            "Shell C (Switzerland) → Integration account. "
            "Each hop crosses jurisdictional boundary."
        ),
    ),
    Scenario(
        scenario_id="T2_02_cross_border_wire",
        name="Cross-Border Wire Transfer",
        description=(
            "4-hop layering scheme crossing 4 jurisdictions via SWIFT wires. "
            "Multiple correspondent banking relationships."
        ),
        injections=[("layering", {"n_hops": 4})],
        tier=2, difficulty="medium", expected_hops=5,
        jurisdictions_used=["US", "UK", "CH", "SG"],
        correct_SAR_elements=[
            "cross_border", "swift_transfer", "multi_jurisdiction", "layering",
        ],
        ground_truth_causal_chain=(
            "Source → 4 shell hops across US/UK/CH/SG → Sink account. "
            "SWIFT transfers on final hop. Amount decays ~3% per hop."
        ),
    ),
    Scenario(
        scenario_id="T2_03_crypto_mixing",
        name="Crypto Mixing Service",
        description=(
            "4-exchange crypto mixing chain. Harder to trace due to "
            "multiple crypto-to-crypto transfers."
        ),
        injections=[("crypto_mixing", {"n_hops": 4})],
        tier=2, difficulty="medium", expected_hops=5,
        jurisdictions_used=["MT", "SG", "HK", "US"],
        correct_SAR_elements=[
            "crypto_mixing", "exchange_chain", "obfuscation",
        ],
        ground_truth_causal_chain=(
            "Source → Crypto exchange 1 → Exchange 2 → Exchange 3 → "
            "Exchange 4 → Clean account. 4 exchange hops with amount variance."
        ),
    ),
    Scenario(
        scenario_id="T2_04_real_estate",
        name="Real Estate Integration",
        description=(
            "Shell company chain (3 shells) used to purchase real estate. "
            "Integration phase of money laundering cycle."
        ),
        injections=[("shell_company", {"n_shells": 3})],
        tier=2, difficulty="medium", expected_hops=4,
        jurisdictions_used=["PA", "KY", "US"],
        correct_SAR_elements=[
            "real_estate", "shell_ownership", "integration_phase", "high_value",
        ],
        ground_truth_causal_chain=(
            "Individual → Shell A (Panama) → Shell B (Cayman) → "
            "Shell C (US) → Real estate account. "
            "Beneficial ownership obscured through shell layers."
        ),
    ),
    Scenario(
        scenario_id="T2_05_nested_shells",
        name="Nested Shell Companies",
        description=(
            "4 nested shell companies with cross-ownership. "
            "Complex corporate structure to obscure beneficial ownership."
        ),
        injections=[("shell_company", {"n_shells": 4})],
        tier=2, difficulty="medium", expected_hops=5,
        jurisdictions_used=["CY", "MT", "LU", "IE"],
        correct_SAR_elements=[
            "nested_shells", "cross_ownership", "EU_jurisdictions", "complex_structure",
        ],
        ground_truth_causal_chain=(
            "Individual → Shell A (Cyprus) → Shell B (Malta) → "
            "Shell C (Luxembourg) → Shell D (Ireland) → Clean account. "
            "EU regulatory arbitrage through nested corporate structures."
        ),
    ),
    Scenario(
        scenario_id="T2_06_four_hop_multi_jur",
        name="4-Hop Multi-Jurisdiction Layering",
        description=(
            "4-hop layering deliberately routed through jurisdictions with "
            "weak information-sharing agreements."
        ),
        injections=[("layering", {"n_hops": 4})],
        tier=2, difficulty="medium", expected_hops=5,
        jurisdictions_used=["BS", "BZ", "PA", "VG"],
        correct_SAR_elements=[
            "jurisdiction_shopping", "weak_AML", "multi_hop", "offshore",
        ],
        ground_truth_causal_chain=(
            "Source → 4 shell hops through Bahamas/Belize/Panama/BVI → "
            "Sink. Each jurisdiction chosen for weak AML enforcement."
        ),
    ),
    Scenario(
        scenario_id="T2_07_false_positive_trap",
        name="False Positive Trap",
        description=(
            "Small 3-smurf structuring scheme alongside legitimate-looking "
            "traffic. Tests investigator's precision — must avoid flagging "
            "innocent entities."
        ),
        injections=[("smurfing", {"n_smurfs": 3})],
        tier=2, difficulty="medium", expected_hops=2,
        jurisdictions_used=["US"],
        correct_SAR_elements=[
            "structuring", "false_positive_risk", "precision_test",
        ],
        ground_truth_causal_chain=(
            "Individual → 3 smurf accounts → Aggregation. "
            "Small scheme buried in dense legitimate traffic. "
            "Investigator must avoid over-flagging neighbors."
        ),
    ),
    Scenario(
        scenario_id="T2_08_loan_back",
        name="Loan-Back Scheme",
        description=(
            "Funds layered through 3 shell hops, then returned as a "
            "'loan' to the original entity via a different path."
        ),
        injections=[("layering", {"n_hops": 3})],
        tier=2, difficulty="medium", expected_hops=4,
        jurisdictions_used=["CH", "LU", "US"],
        correct_SAR_elements=[
            "loan_back", "circular_flow", "self_lending", "layering",
        ],
        ground_truth_causal_chain=(
            "Source → 3 shell intermediaries → Sink. "
            "Funds appear as legitimate loan from final shell. "
            "Circular flow creates false paper trail."
        ),
    ),
    Scenario(
        scenario_id="T2_09_dividend_stripping",
        name="Dividend Stripping",
        description=(
            "Trade-based scheme using 3 intermediaries to disguise "
            "payments as dividend distributions from shell companies."
        ),
        injections=[("trade_based", {"n_intermediaries": 3})],
        tier=2, difficulty="medium", expected_hops=4,
        jurisdictions_used=["CY", "MT", "NL"],
        correct_SAR_elements=[
            "dividend_stripping", "tax_evasion", "corporate_distributions",
        ],
        ground_truth_causal_chain=(
            "Shell company → 3 intermediaries (disguised as dividends) → "
            "Sink. Trade finance transactions mispriced as corporate dividends."
        ),
    ),
    Scenario(
        scenario_id="T2_10_trade_mispricing",
        name="Trade Mispricing",
        description=(
            "4-intermediary trade-based scheme with systematic "
            "over-invoicing to move value across borders."
        ),
        injections=[("trade_based", {"n_intermediaries": 4})],
        tier=2, difficulty="medium", expected_hops=5,
        jurisdictions_used=["HK", "SG", "US", "UK"],
        correct_SAR_elements=[
            "trade_mispricing", "over_invoicing", "cross_border_trade",
        ],
        ground_truth_causal_chain=(
            "Shell → 4 intermediaries (systematic over-invoicing) → Sink. "
            "Invoice values consistently 20-30% above fair market across hops."
        ),
    ),
]


# ---------------------------------------------------------------------------
# Tier 3 — Hard (8 scenarios)
# ---------------------------------------------------------------------------

TIER_3_HARD: List[Scenario] = [
    Scenario(
        scenario_id="T3_01_six_hop_multi_jur",
        name="6-Hop Multi-Jurisdiction Layering",
        description=(
            "6-hop layering chain crossing 6 distinct jurisdictions. "
            "Maximum obfuscation through jurisdictional complexity."
        ),
        injections=[("layering", {"n_hops": 6})],
        tier=3, difficulty="hard", expected_hops=7,
        jurisdictions_used=["US", "UK", "CH", "SG", "HK", "KY"],
        correct_SAR_elements=[
            "deep_layering", "multi_jurisdiction", "complex_trail",
            "correspondent_banking", "amount_decay",
        ],
        ground_truth_causal_chain=(
            "Source → 6 shell hops across 6 jurisdictions → Sink. "
            "Amount decays ~3%/hop. SWIFT on final leg. "
            "Maximum jurisdictional complexity requires full cross-referencing."
        ),
    ),
    Scenario(
        scenario_id="T3_02_dual_scheme_shared_nodes",
        name="Two Simultaneous Schemes, Shared Nodes",
        description=(
            "Smurfing AND layering share intermediate nodes. "
            "Investigator must disentangle two overlapping schemes."
        ),
        injections=[
            ("smurfing", {"n_smurfs": 5}),
            ("layering", {"n_hops": 4}),
        ],
        tier=3, difficulty="hard", expected_hops=6,
        num_schemes=2,
        jurisdictions_used=["US", "CH", "KY", "PA"],
        correct_SAR_elements=[
            "dual_scheme", "shared_infrastructure", "smurfing", "layering",
            "scheme_entanglement",
        ],
        ground_truth_causal_chain=(
            "Scheme A: Individual → 5 smurfs → Aggregation. "
            "Scheme B: Account → 4 shell hops → Sink. "
            "Shared intermediate nodes create confusion. "
            "Investigator must file SAR covering both schemes."
        ),
    ),
    Scenario(
        scenario_id="T3_03_circular_ownership",
        name="8-Hop Circular Ownership",
        description=(
            "8-hop layering creating near-circular ownership structure "
            "where beneficial owner is obscured behind 8 layers."
        ),
        injections=[("layering", {"n_hops": 8})],
        tier=3, difficulty="hard", expected_hops=9,
        jurisdictions_used=["KY", "VG", "BS", "PA", "BZ", "CY", "MT", "LU"],
        correct_SAR_elements=[
            "circular_ownership", "deep_layering", "8_hops",
            "beneficial_owner_hidden", "offshore_chain",
        ],
        ground_truth_causal_chain=(
            "Source → 8 shell layers across 8 offshore jurisdictions → Sink. "
            "Near-circular structure makes it appear each entity owns the next. "
            "Maximum tracing depth required."
        ),
    ),
    Scenario(
        scenario_id="T3_04_correspondent_banking",
        name="Correspondent Banking Abuse",
        description=(
            "5-hop layering exploiting correspondent banking relationships. "
            "Funds move through SWIFT network via nostro/vostro accounts."
        ),
        injections=[("layering", {"n_hops": 5})],
        tier=3, difficulty="hard", expected_hops=6,
        jurisdictions_used=["US", "UK", "DE", "FR", "CH"],
        correct_SAR_elements=[
            "correspondent_banking", "SWIFT_network", "nostro_vostro",
            "cross_border", "deep_layering",
        ],
        ground_truth_causal_chain=(
            "Source → 5 shell intermediaries via correspondent banks → Sink. "
            "Each hop uses SWIFT through major banking centers. "
            "Correspondent relationships obscure ultimate beneficiary."
        ),
    ),
    Scenario(
        scenario_id="T3_05_complex_false_positive",
        name="Complex False Positive Scenario",
        description=(
            "Large 8-smurf structuring scheme embedded in extremely dense "
            "transaction neighborhood. Maximum false positive risk."
        ),
        injections=[("smurfing", {"n_smurfs": 8})],
        tier=3, difficulty="hard", expected_hops=2,
        jurisdictions_used=["US"],
        correct_SAR_elements=[
            "structuring", "dense_neighborhood", "false_positive_defense",
            "precision_critical",
        ],
        ground_truth_causal_chain=(
            "Individual → 8 smurf accounts → Aggregation. "
            "Dense transaction neighborhood means many innocent entities "
            "appear suspicious. Investigator must maintain high precision."
        ),
    ),
    Scenario(
        scenario_id="T3_06_mirror_trading",
        name="Mirror Trading",
        description=(
            "Simultaneous crypto mixing and trade-based schemes that "
            "mirror each other's transaction patterns."
        ),
        injections=[
            ("crypto_mixing", {"n_hops": 4}),
            ("trade_based", {"n_intermediaries": 4}),
        ],
        tier=3, difficulty="hard", expected_hops=9,
        num_schemes=2,
        jurisdictions_used=["MT", "SG", "HK", "US", "UK"],
        correct_SAR_elements=[
            "mirror_trading", "dual_channel", "crypto_and_trade",
            "parallel_schemes", "correlated_timing",
        ],
        ground_truth_causal_chain=(
            "Scheme A: Account → 4 crypto exchanges → Clean account. "
            "Scheme B: Shell → 4 intermediaries (trade) → Sink. "
            "Mirror transaction patterns across crypto and trade channels."
        ),
    ),
    Scenario(
        scenario_id="T3_07_layering_with_mules",
        name="Layering with Mule Network",
        description=(
            "Large smurfing network (10 mules) combined with 5-hop "
            "layering. Cash collected from mules then layered offshore."
        ),
        injections=[
            ("smurfing", {"n_smurfs": 10}),
            ("layering", {"n_hops": 5}),
        ],
        tier=3, difficulty="hard", expected_hops=7,
        num_schemes=2,
        jurisdictions_used=["US", "UK", "KY", "PA", "CH"],
        correct_SAR_elements=[
            "mule_network", "smurfing", "layering", "combined_scheme",
            "placement_and_layering",
        ],
        ground_truth_causal_chain=(
            "Scheme A: Individual → 10 mule accounts → Aggregation (placement). "
            "Scheme B: Account → 5 shell hops → Offshore sink (layering). "
            "Mule network feeds into layering pipeline."
        ),
    ),
    Scenario(
        scenario_id="T3_08_real_estate_multi_country",
        name="Real Estate Across 3 Countries",
        description=(
            "4-shell chain investing in real estate across 3 countries. "
            "Integration phase with cross-border property purchases."
        ),
        injections=[("shell_company", {"n_shells": 4})],
        tier=3, difficulty="hard", expected_hops=5,
        jurisdictions_used=["US", "UK", "FR", "KY"],
        correct_SAR_elements=[
            "real_estate", "multi_country", "shell_ownership",
            "cross_border_property", "integration",
        ],
        ground_truth_causal_chain=(
            "Individual → 4 shell companies across KY/US/UK/FR → "
            "Integration account. Shell layers used to purchase "
            "real estate in 3 countries simultaneously."
        ),
    ),
]


# ---------------------------------------------------------------------------
# Tier 4 — Extreme (4 placeholders for Criminal Codex)
# ---------------------------------------------------------------------------

TIER_4_EXTREME: List[Scenario] = [
    Scenario(
        scenario_id="T4_01_codex_placeholder",
        name="Criminal Codex Scheme #1",
        description="Reserved for Criminal Codex AI-generated scheme. "
                    "Will be populated during training by the Criminal Designer agent.",
        injections=[("layering", {"n_hops": 6})],  # default injection, replaced at runtime
        tier=4, difficulty="extreme", expected_hops=7,
        jurisdictions_used=["US", "CH", "KY", "SG", "HK", "MT"],
        is_codex_generated=True,
        correct_SAR_elements=["codex_generated", "novel_pattern"],
        ground_truth_causal_chain="Criminal Codex generated — pattern unknown at design time.",
    ),
    Scenario(
        scenario_id="T4_02_codex_placeholder",
        name="Criminal Codex Scheme #2",
        description="Reserved for Criminal Codex AI-generated scheme. "
                    "Novel combination of existing scheme types.",
        injections=[("crypto_mixing", {"n_hops": 5})],
        tier=4, difficulty="extreme", expected_hops=6,
        jurisdictions_used=["MT", "SG", "HK", "US", "UK", "EE"],
        is_codex_generated=True,
        correct_SAR_elements=["codex_generated", "novel_combination"],
        ground_truth_causal_chain="Criminal Codex generated — pattern unknown at design time.",
    ),
    Scenario(
        scenario_id="T4_03_codex_placeholder",
        name="Criminal Codex Scheme #3",
        description="Reserved for Criminal Codex AI-generated scheme. "
                    "Adversarial scheme designed to exploit investigator weaknesses.",
        injections=[("trade_based", {"n_intermediaries": 5})],
        tier=4, difficulty="extreme", expected_hops=6,
        jurisdictions_used=["HK", "SG", "PA", "BS", "KY", "VG"],
        is_codex_generated=True,
        correct_SAR_elements=["codex_generated", "adversarial_design"],
        ground_truth_causal_chain="Criminal Codex generated — adversarial exploitation.",
    ),
    Scenario(
        scenario_id="T4_04_codex_placeholder",
        name="Criminal Codex Scheme #4 (Zero-Day)",
        description="Reserved for Zero-Day Reveal — the criminal's best unconstrained "
                    "scheme. Novel AI-invented crime pattern for demo closing moment.",
        injections=[
            ("smurfing", {"n_smurfs": 8}),
            ("layering", {"n_hops": 6}),
        ],
        tier=4, difficulty="extreme", expected_hops=8,
        num_schemes=2,
        jurisdictions_used=["US", "UK", "CH", "SG", "HK", "KY", "PA", "MT"],
        is_codex_generated=True,
        correct_SAR_elements=["codex_generated", "zero_day", "novel_pattern"],
        ground_truth_causal_chain="Zero-Day Reveal — AI-invented crime pattern. "
                                  "Graph edit distance confirms novelty.",
    ),
]


# ---------------------------------------------------------------------------
# Full library
# ---------------------------------------------------------------------------

ALL_SCENARIOS: List[Scenario] = TIER_1_EASY + TIER_2_MEDIUM + TIER_3_HARD + TIER_4_EXTREME

SCENARIO_BY_ID: Dict[str, Scenario] = {s.scenario_id: s for s in ALL_SCENARIOS}

SCENARIOS_BY_TIER: Dict[int, List[Scenario]] = {
    1: TIER_1_EASY,
    2: TIER_2_MEDIUM,
    3: TIER_3_HARD,
    4: TIER_4_EXTREME,
}

SCENARIOS_BY_DIFFICULTY: Dict[str, List[Scenario]] = {
    "easy":    TIER_1_EASY,
    "medium":  TIER_2_MEDIUM,
    "hard":    TIER_3_HARD,
    "extreme": TIER_4_EXTREME,
}


# ---------------------------------------------------------------------------
# Scenario selection helpers
# ---------------------------------------------------------------------------

def get_scenario(scenario_id: str) -> Scenario:
    """Get a scenario by ID. Raises KeyError if not found."""
    return SCENARIO_BY_ID[scenario_id]


def get_scenarios_by_tier(tier: int) -> List[Scenario]:
    """Get all scenarios for a tier (1-4)."""
    return SCENARIOS_BY_TIER.get(tier, [])


def get_random_scenario(
    tier: Optional[int] = None,
    rng: Optional[Any] = None,
) -> Scenario:
    """Pick a random scenario, optionally filtered by tier."""
    import numpy as np
    rng = rng or np.random.default_rng()
    pool = get_scenarios_by_tier(tier) if tier else ALL_SCENARIOS
    if not pool:
        raise ValueError(f"No scenarios for tier={tier}")
    idx = int(rng.integers(0, len(pool)))
    return pool[idx]


def inject_scenario(
    graph: TransactionGraph,
    scenario: Scenario,
) -> List[Dict[str, Any]]:
    """
    Inject a scenario into the graph and return ground truth.

    This is the primary entry point for the environment.
    Returns list of ground_truth dicts (one per scheme in the scenario).
    """
    return scenario.inject(graph)


def inject_random_scenario(
    graph: TransactionGraph,
    tier: Optional[int] = None,
    rng: Optional[Any] = None,
) -> Tuple[Scenario, List[Dict[str, Any]]]:
    """
    Pick a random scenario and inject it.
    Returns (scenario, [ground_truth_dicts]).
    """
    scenario = get_random_scenario(tier=tier, rng=rng)
    results = inject_scenario(graph, scenario)
    return scenario, results


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_all_scenarios(seed: int = 42) -> Dict[str, Any]:
    """
    Inject all 30 scenarios one at a time into a fresh graph.
    Verifies each scenario injects cleanly, produces valid ground truth,
    and resets cleanly.

    Returns summary dict.
    """
    import time
    results: Dict[str, Any] = {
        "total": len(ALL_SCENARIOS),
        "passed": 0,
        "failed": 0,
        "errors": [],
        "timing": {},
    }

    tg = TransactionGraph(seed=seed)
    base_nodes = tg.graph.number_of_nodes()
    base_edges = tg.graph.number_of_edges()

    for scenario in ALL_SCENARIOS:
        t0 = time.time()
        try:
            # Inject
            gts = scenario.inject(tg)

            # Verify ground truth populated
            for gt in gts:
                assert "source_entity" in gt, f"Missing source_entity"
                assert "sink_entity" in gt, f"Missing sink_entity"
                assert "full_path" in gt, f"Missing full_path"
                assert len(gt["full_path"]) >= 2, f"Path too short: {gt['full_path']}"
                assert gt["scenario_id"] == scenario.scenario_id
                assert gt["difficulty"] == scenario.difficulty
                assert gt["tier"] == scenario.tier
                assert gt.get("is_codex_generated") == scenario.is_codex_generated

            # Verify edges exist in graph
            for sid in list(tg._injected.keys()):
                for rec in tg._injected[sid]["edges"]:
                    assert tg.graph.has_edge(rec["src"], rec["tgt"]), \
                        f"Missing edge: {rec['src']} → {rec['tgt']}"

            # Reset and verify clean
            tg.reset()
            assert len(tg._injected) == 0, "Injected schemes not cleared"
            assert len(tg.ground_truth) == 0, "Ground truth not cleared"
            assert tg.graph.number_of_nodes() == base_nodes, \
                f"Node count mismatch after reset: {tg.graph.number_of_nodes()} != {base_nodes}"

            elapsed = time.time() - t0
            results["passed"] += 1
            results["timing"][scenario.scenario_id] = round(elapsed, 4)

        except Exception as e:
            tg.reset()
            results["failed"] += 1
            results["errors"].append({
                "scenario_id": scenario.scenario_id,
                "error": str(e),
            })

    return results


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time

    print("=" * 70)
    print("HEIST Scenario Library — Validation")
    print("=" * 70)

    print(f"\nTotal scenarios: {len(ALL_SCENARIOS)}")
    for tier in [1, 2, 3, 4]:
        scenarios = get_scenarios_by_tier(tier)
        label = {1: "Easy", 2: "Medium", 3: "Hard", 4: "Extreme"}[tier]
        print(f"  Tier {tier} ({label}): {len(scenarios)} scenarios")

    print(f"\nValidating all {len(ALL_SCENARIOS)} scenarios...")
    t0 = time.time()
    result = validate_all_scenarios(seed=77)
    elapsed = time.time() - t0

    print(f"\n  Passed: {result['passed']}/{result['total']}")
    print(f"  Failed: {result['failed']}")
    print(f"  Time:   {elapsed:.2f}s")

    if result["errors"]:
        print("\n  ERRORS:")
        for err in result["errors"]:
            print(f"    {err['scenario_id']}: {err['error']}")

    print(f"\n  Scenario Details:")
    for s in ALL_SCENARIOS:
        codex = " [CODEX]" if s.is_codex_generated else ""
        multi = f" [{s.num_schemes} schemes]" if s.num_schemes > 1 else ""
        print(f"    {s.scenario_id:35s} | {s.difficulty:8s} | "
              f"{s.expected_hops} hops | {s.name}{codex}{multi}")

    if result["failed"] == 0:
        print(f"\n  ALL {result['total']} SCENARIOS VALIDATED SUCCESSFULLY")
    else:
        print(f"\n  VALIDATION FAILED — {result['failed']} errors")
        exit(1)
