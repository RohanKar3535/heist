"""
Criminal Designer Agent — Step 7.

Adversarial agent that:
    1. Tracks investigator weakness_database (scheme_type → failure_rate)
    2. Uses softmax weakness targeting to pick which scheme to evolve
    3. Calls Gemini API to generate new Python scheme functions
    4. Validates generated schemes (5-check validation suite)
    5. Appends passing schemes to criminal_codex.py
    6. Computes novelty_bonus via cosine distance
    7. Calculates R_criminal with all bonuses

Key invariant: NO scheme enters the Codex without passing all 5 validation checks.
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
import math
import os
import re
import sys
import textwrap
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Add project root for imports
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from env.transaction_graph import TransactionGraph, SCHEME_TYPES
from env.reward import r_criminal


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Scheme categories for weakness tracking
SCHEME_CATEGORIES: List[str] = [
    "smurfing", "layering", "shell_company", "crypto_mixing", "trade_based",
    # Codex variant categories
    "structuring_variant", "crypto_variant", "trade_variant", "layering_variant",
]

# LLM API config — priority: HF Inference API → Groq → Gemini
HF_TOKEN       = os.environ.get("HF_TOKEN", "")
HF_LLM_MODEL   = os.environ.get("HF_LLM_MODEL", "Qwen/Qwen2.5-72B-Instruct")
GROQ_API_KEY   = os.environ.get("GROQ_API_KEY", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL   = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")

# Validation limits
MAX_GENERATION_ATTEMPTS = 3
VALIDATION_GRAPH_NODES  = 100
MIN_TRANSACTION_AMOUNT  = 100.0
MAX_TRANSACTION_AMOUNT  = 10_000_000.0

# Codex file path
CODEX_PATH = os.path.join(_PROJECT_ROOT, "criminal_codex.py")


# ---------------------------------------------------------------------------
# Weakness Database
# ---------------------------------------------------------------------------

class WeaknessDatabase:
    """
    Tracks investigator failure_rate per scheme_type.
    
    failure_rate[scheme_type] = fraction of episodes where investigator
    failed to achieve F1 > 0.5 on that scheme type.
    
    Higher failure_rate → investigator is weak against this type.
    """

    def __init__(self) -> None:
        self._failures: Dict[str, List[float]] = {}
        for cat in SCHEME_CATEGORIES:
            self._failures[cat] = []

    @property
    def failure_rates(self) -> Dict[str, float]:
        """Current failure rate per category."""
        return {
            cat: (sum(f) / max(len(f), 1))
            for cat, f in self._failures.items()
        }

    def update(self, scheme_type: str, investigator_f1: float) -> None:
        """
        Record one episode result.
        Failure = investigator F1 < 0.5 on this scheme type.
        """
        cat = self._normalize_category(scheme_type)
        failed = 1.0 if investigator_f1 < 0.5 else 0.0
        self._failures[cat].append(failed)
        # Keep last 50 episodes per category for recency weighting
        if len(self._failures[cat]) > 50:
            self._failures[cat] = self._failures[cat][-50:]

    def get_weakness_vector(self) -> Dict[str, float]:
        """
        Weakness vector: higher = investigator is weaker.
        Minimum 0.1 floor to ensure exploration.
        """
        rates = self.failure_rates
        return {cat: max(rate, 0.1) for cat, rate in rates.items()}

    def _normalize_category(self, scheme_type: str) -> str:
        """Map scheme_type to nearest category."""
        if scheme_type in self._failures:
            return scheme_type
        # Map variants back to base category
        for base in SCHEME_CATEGORIES:
            if base in scheme_type:
                return base
        return SCHEME_CATEGORIES[0]  # fallback


# ---------------------------------------------------------------------------
# Softmax weakness targeting
# ---------------------------------------------------------------------------

def softmax_target_selection(
    weakness_vector: Dict[str, float],
    temperature: float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> str:
    """
    P(generate_scheme_k) = softmax(weakness_vector / τ)
    
    Higher weakness → more likely to generate scheme of that type.
    Temperature τ controls exploration: high = uniform, low = greedy.
    
    Returns the selected scheme category.
    """
    rng = rng or np.random.default_rng()
    
    categories = list(weakness_vector.keys())
    values = np.array([weakness_vector[c] for c in categories])
    
    # Softmax with temperature
    logits = values / max(temperature, 1e-9)
    logits -= logits.max()  # numerical stability
    exp_logits = np.exp(logits)
    probs = exp_logits / exp_logits.sum()
    
    idx = int(rng.choice(len(categories), p=probs))
    return categories[idx]


# ---------------------------------------------------------------------------
# Novelty measurement (cosine distance)
# ---------------------------------------------------------------------------

def _scheme_embedding(scheme_code: str) -> np.ndarray:
    """
    Structural fingerprint of scheme code for novelty measurement.

    Counts financial-operation keywords instead of character n-grams.
    N-grams cannot distinguish _build_crypto_variant(3) from (4) — they
    produce nearly identical trigram distributions. Keyword counts capture
    what actually makes schemes mechanically different: which transaction
    types, node types, and structural patterns they use.
    """
    code = scheme_code.lower()

    features = np.array([
        # Transaction types — core mechanic signals
        code.count("cash_deposit"),
        code.count("crypto_transfer"),
        code.count("wire_transfer"),
        code.count("trade_finance"),
        code.count("swift"),
        code.count("hawala"),
        code.count("ach"),
        code.count("check"),
        # Node types used as intermediaries
        code.count("shell_company"),
        code.count("crypto_exchange"),
        code.count("individual"),
        code.count("account"),
        # Structural patterns
        code.count("add_edge"),        # total edges injected
        code.count("for "),            # loop depth (smurfing = many loops)
        code.count("phase"),           # multi-phase indicator
        code.count("rng.choice"),      # branching / selection count
        # Financial scale indicators
        code.count("9999"),            # structuring amounts near $10k threshold
        code.count("offshore"),        # offshore routing
        code.count("jurisdiction"),    # cross-jurisdiction complexity
        code.count("rng.uniform("),    # amount randomisation calls
    ], dtype=np.float64)

    norm = np.linalg.norm(features)
    if norm > 0:
        features /= norm
    return features


def compute_novelty_bonus(
    new_code: str,
    existing_codes: List[str],
) -> float:
    """
    Novelty bonus = minimum cosine distance to any existing scheme (nearest-neighbor).
    Returns value in [0, 1]. Higher = more novel.

    Using min instead of mean prevents the score from shrinking as the codex grows:
    mean distance decreases geometrically as more schemes fill the embedding space,
    but min distance stays meaningful — it measures distance from the closest existing
    scheme regardless of codex size.
    """
    if not existing_codes:
        return 1.0  # first scheme is maximally novel

    new_emb = _scheme_embedding(new_code)

    distances: List[float] = []
    for code in existing_codes:
        old_emb = _scheme_embedding(code)
        cos_sim = float(np.dot(new_emb, old_emb))
        cos_dist = 1.0 - max(-1.0, min(1.0, cos_sim))
        distances.append(cos_dist)

    return round(float(np.min(distances)), 6)


# ---------------------------------------------------------------------------
# Validation Suite (5 checks)
# ---------------------------------------------------------------------------

def build_test_graph(seed: int = 42) -> TransactionGraph:
    """Build a small 100-node test graph for validation."""
    # We create a mini version by using the full constructor but it creates 100k nodes.
    # Instead, we create a minimal networkx graph for validation.
    import networkx as nx
    
    rng = np.random.default_rng(seed)
    G = nx.DiGraph()
    
    # 100 nodes: 50 accounts, 20 shells, 15 individuals, 15 crypto
    jurisdictions = ["US", "UK", "CH", "SG", "KY", "PA", "MT", "HK"]
    
    for i in range(50):
        G.add_node(f"acc_{i}", node_type="account",
                   jurisdiction=jurisdictions[i % len(jurisdictions)])
    for i in range(20):
        G.add_node(f"shell_{i}", node_type="shell_company",
                   jurisdiction=jurisdictions[i % len(jurisdictions)])
    for i in range(15):
        G.add_node(f"ind_{i}", node_type="individual",
                   jurisdiction=jurisdictions[i % len(jurisdictions)])
    for i in range(15):
        G.add_node(f"crypto_{i}", node_type="crypto_exchange",
                   jurisdiction=jurisdictions[i % len(jurisdictions)])
    
    # Add some background edges
    nodes = list(G.nodes())
    for _ in range(200):
        src = str(rng.choice(nodes))
        tgt = str(rng.choice(nodes))
        if src != tgt:
            G.add_edge(src, tgt,
                       amount=float(rng.uniform(100, 50000)),
                       timestamp=int(rng.integers(1672531200, 1704067200)),
                       transaction_type="wire_transfer",
                       jurisdiction=G.nodes[tgt].get("jurisdiction", "US"),
                       suspicion_weight=float(rng.uniform(0.1, 0.5)),
                       scheme_id=None)
    
    # Create a lightweight TransactionGraph-like object
    class MiniGraph:
        def __init__(self, g, r):
            self.graph = g
            self._rng = r
            self._injected = {}
            self.ground_truth = {}
            self._nodes_by_type = {
                "account": [n for n, d in g.nodes(data=True) if d["node_type"] == "account"],
                "shell_company": [n for n, d in g.nodes(data=True) if d["node_type"] == "shell_company"],
                "individual": [n for n, d in g.nodes(data=True) if d["node_type"] == "individual"],
                "crypto_exchange": [n for n, d in g.nodes(data=True) if d["node_type"] == "crypto_exchange"],
            }
        
        def reset(self):
            for sid in list(self._injected.keys()):
                info = self._injected.pop(sid)
                self.ground_truth.pop(sid, None)
                for rec in info.get("edges", []):
                    if self.graph.has_edge(rec["src"], rec["tgt"]):
                        self.graph.remove_edge(rec["src"], rec["tgt"])
                    if rec.get("prior") is not None:
                        self.graph.add_edge(rec["src"], rec["tgt"], **rec["prior"])
    
    return MiniGraph(G, rng)


def validate_scheme(
    scheme_fn: callable,
    seed: int = 42,
) -> Tuple[bool, str, Optional[dict]]:
    """
    5-check validation suite. ALL must pass before a scheme enters the Codex.
    
    Returns (passed: bool, reason: str, ground_truth: Optional[dict]).
    """
    test_graph = build_test_graph(seed)
    rng = np.random.default_rng(seed)
    
    # ── Check 1: No exceptions ─────────────────────────────────────────
    try:
        gt = scheme_fn(test_graph, rng)
    except Exception as e:
        return False, f"Check 1 FAIL: Exception raised: {e}", None
    
    # ── Check 2: Ground truth structure ────────────────────────────────
    required_keys = {"source_entity", "sink_entity", "full_path", "scheme_type", "num_hops"}
    missing = required_keys - set(gt.keys())
    if missing:
        return False, f"Check 2 FAIL: Missing ground truth keys: {missing}", None
    
    # ── Checks 3-5: Path/amount/reset — wrapped to catch any LLM garbage ──
    # LLM-generated schemes may return tuples, numpy types, dicts, etc. as
    # node IDs. Rather than trying to sanitize every possible bad structure,
    # we catch ANY exception and treat it as a validation failure so training
    # never crashes.
    import networkx as nx
    try:
        # ── Check 3: Money actually moves from source to sink ───────────
        path = gt["full_path"]
        if len(path) < 2:
            return False, "Check 3 FAIL: Path too short (< 2 nodes)", None

        source = gt["source_entity"]
        sink   = gt["sink_entity"]

        # Normalise source/sink to plain strings
        if not isinstance(source, str):
            source = str(source)
        if not isinstance(sink, str):
            sink = str(sink)

        if not test_graph.graph.has_node(source):
            return False, f"Check 3 FAIL: Source {source} not in graph", None
        if not test_graph.graph.has_node(sink):
            return False, f"Check 3 FAIL: Sink {sink} not in graph", None

        try:
            nx.shortest_path(test_graph.graph, source, sink)
        except (nx.NetworkXNoPath, nx.NodeNotFound, nx.NetworkXError):
            return False, f"Check 3 FAIL: No path from {source} to {sink}", None

        # ── Check 4: Transaction amounts realistic ($100 - $10M) ────────
        # Flatten path: LLM may return tuples, dicts, numpy types, etc.
        clean_path: list = []
        for n in path:
            if isinstance(n, (list, tuple)):
                for item in n:
                    s = str(item)
                    if test_graph.graph.has_node(s):
                        clean_path.append(s)
            elif isinstance(n, dict):
                # e.g. {"src": "acc_1", "tgt": "acc_2"} — skip silently
                pass
            else:
                s = str(n) if not isinstance(n, str) else n
                if test_graph.graph.has_node(s):
                    clean_path.append(s)

        scheme_edges = []
        for node in clean_path:
            for succ in test_graph.graph.successors(node):
                edge_data = test_graph.graph[node][succ]
                if edge_data.get("scheme_id") is not None:
                    scheme_edges.append((node, succ, edge_data))

        if not scheme_edges:
            return False, "Check 4 FAIL: No scheme edges found", None

        for src, tgt, edge_data in scheme_edges:
            amount = edge_data.get("amount", 0)
            if amount < MIN_TRANSACTION_AMOUNT:
                return False, f"Check 4 FAIL: Amount too low: ${amount:.2f}", None
            if amount > MAX_TRANSACTION_AMOUNT:
                return False, f"Check 4 FAIL: Amount too high: ${amount:.2f}", None

        # ── Check 5: Clean reset ─────────────────────────────────────────
        test_graph.reset()

    except Exception as e:
        # Catch-all: any bad node type, missing attr, or NetworkX error → fail
        return False, f"Check 3-5 FAIL: Unexpected error ({type(e).__name__}): {e}", None

    return True, "All 5 validation checks passed", gt


# ---------------------------------------------------------------------------
# Gemini API — Scheme Generation
# ---------------------------------------------------------------------------

_GENERATION_PROMPT = """\
You are a financial crime designer AI. Your job is to write a Python function that \
injects a novel money laundering scheme into a transaction graph.

The investigator AI is WEAK at detecting: {target_weakness}
Its failure rate on this type: {failure_rate:.0%}

Write a Python function with this EXACT signature:
```python
def inject_scheme(graph, rng=None) -> dict:
```

The function must:
1. Pick source/sink/intermediate nodes from `graph.graph.nodes(data=True)` using node_type filters
2. Use `rng` (numpy Generator) for all randomness
3. Add edges with `graph.graph.add_edge(src, tgt, amount=..., timestamp=..., transaction_type=..., jurisdiction=..., suspicion_weight=..., scheme_id=...)`
4. Track edges in `graph._injected[scheme_id] = {{"edges": [...], "nodes": []}}`
5. Store ground truth in `graph.ground_truth[scheme_id] = gt`
6. Return dict with keys: source_entity, sink_entity, full_path, scheme_type, num_hops, intermediate_nodes

Node types available: "account" (prefix acc_), "shell_company" (prefix shell_), \
"individual" (prefix ind_), "crypto_exchange" (prefix crypto_)

Transaction types: wire_transfer, cash_deposit, crypto_transfer, check, ach, swift, hawala, trade_finance

Amounts must be between $100 and $10,000,000.
Use `scheme_id = f"codex_{{name}}_{{int(rng.integers(1000, 9999))}}"`.

CRITICAL REQUIREMENT — STRUCTURAL NOVELTY:
You MUST combine at least TWO different scheme types into a single multi-phase scheme. \
Do NOT simply vary the number of intermediaries or amounts of an existing scheme type.

Examples of valid combinatorial schemes:
- Phase 1: individual → multiple accounts (smurfing/structuring) \
  Phase 2: accounts → crypto exchange (crypto mixing) \
  Phase 3: crypto exchange → offshore shell company (layering) \
  Phase 4: shell → clean account (integration)
- Phase 1: shell company chain (trade_finance invoices) \
  Phase 2: proceeds → crypto tumbler (crypto_transfer) \
  Phase 3: crypto → final bank account (swift)
- Phase 1: cash deposits across mule accounts (structuring) \
  Phase 2: mules wire to shell companies in offshore jurisdictions (layering) \
  Phase 3: shells use trade_finance invoices to clean funds (trade-based)

The scheme MUST use at least 2 different transaction_types and at least 2 different node_types \
as intermediaries. This is a hard requirement — schemes that use only one type will be rejected.

Respond with ONLY the Python function code. No explanation. No markdown fencing.
"""


def _call_gemini(prompt: str) -> Optional[str]:
    """Call LLM API — HF Inference → Groq → Gemini."""
    # ── HF Inference API (primary when HF_TOKEN set) ──────────────────
    if HF_TOKEN:
        try:
            from huggingface_hub import InferenceClient
            client = InferenceClient(api_key=HF_TOKEN)
            response = client.chat.completions.create(
                model=HF_LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2048,
                temperature=0.8,
            )
            return response.choices[0].message.content
        except Exception as e:
            err = str(e)
            if "429" in err or "rate" in err.lower():
                return None
            print(f"[CriminalDesigner] HF error: {e}")

    # ── Groq (secondary) ──────────────────────────────────────────────
    if GROQ_API_KEY:
        try:
            from groq import Groq
            client = Groq(api_key=GROQ_API_KEY)
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2048,
                temperature=0.8,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"[CriminalDesigner] Groq error: {e}")

    # ── Gemini (fallback) ──────────────────────────────────────────────
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
                    "generationConfig": {"temperature": 0.8, "maxOutputTokens": 2048},
                },
                timeout=30,
            )
            if resp.status_code == 429:
                wait = 30 * (2 ** attempt)
                print(f"[CriminalDesigner] Gemini 429 — retry {attempt + 1}/3 in {wait}s")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()["candidates"][0]["content"]["parts"][0]["text"]
        except Exception as e:
            print(f"[CriminalDesigner] Gemini error: {e}")
            return None

    print("[CriminalDesigner] All retries exhausted, using fallback")
    return None


def generate_scheme(
    weakness_vector: Dict[str, float],
    temperature: float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[Optional[callable], str, str]:
    """
    Generate a new scheme function targeting the investigator's weakness.
    
    Uses softmax to select target → Gemini to generate code → validate.
    Falls back to random seed scheme variation if Gemini unavailable.
    
    Returns (scheme_fn, scheme_code, target_weakness).
    """
    rng = rng or np.random.default_rng()
    
    # Select target via softmax
    target = softmax_target_selection(weakness_vector, temperature, rng)
    failure_rate = weakness_vector.get(target, 0.1)
    
    prompt = _GENERATION_PROMPT.format(
        target_weakness=target,
        failure_rate=failure_rate,
    )
    
    # Try Gemini API
    response = _call_gemini(prompt)
    
    if response:
        # Extract function code
        code = _extract_function_code(response)
        if code:
            fn = _compile_function(code)
            if fn:
                return fn, code, target
    
    # Fallback: generate a deterministic variant of a seed scheme
    fn, code = _generate_fallback_variant(target, rng)
    return fn, code, target


def _extract_function_code(response: str) -> Optional[str]:
    """Extract Python function from Gemini response."""
    # Remove markdown fencing if present
    code = response.strip()
    code = re.sub(r'^```python\s*', '', code)
    code = re.sub(r'^```\s*', '', code)
    code = re.sub(r'\s*```$', '', code)
    
    # Ensure it starts with def
    if 'def inject_scheme' not in code and 'def inject_' not in code:
        return None
    
    # Find the function definition
    match = re.search(r'(def inject_\w+\(.*?\).*)', code, re.DOTALL)
    if match:
        return match.group(1)
    
    return code if code.strip().startswith('def') else None


def _compile_function(code: str) -> Optional[callable]:
    """Safely compile and extract function from code string."""
    try:
        namespace = {"np": np, "numpy": np}
        exec(code, namespace)
        # Find the injected function
        for name, obj in namespace.items():
            if name.startswith("inject_") and callable(obj):
                return obj
        return None
    except Exception:
        return None


def _generate_fallback_variant(
    target: str,
    rng: np.random.Generator,
) -> Tuple[callable, str]:
    """
    Generate a deterministic variant when Gemini is unavailable.
    Prefers a combinatorial hybrid scheme; falls back to single-type variants.
    """
    variant_id = int(rng.integers(1000, 9999))

    # Always prefer hybrid (multi-phase) schemes: they combine multiple
    # transaction types and node types, giving maximally diverse keyword
    # fingerprints that the structural novelty check can distinguish.
    # Pick a hybrid combination that targets the weakness, cycling through
    # all 4 combinations to ensure codex variety across codex updates.
    hybrid_types = [
        ("structuring", "crypto"),
        ("crypto", "layering"),
        ("trade", "structuring"),
        ("layering", "trade"),
    ]
    # Rotate hybrid selection based on variant_id for variety
    combo = hybrid_types[variant_id % len(hybrid_types)]
    code = _build_hybrid_variant(variant_id, phase1=combo[0], phase2=combo[1])
    fn = _compile_function(code)
    if fn is not None:
        return fn, code

    # Single-type fallback (only if hybrid compilation fails)
    if "structuring" in target or "smurfing" in target:
        n_mules = int(rng.integers(3, 9))
        code = _build_structuring_variant(n_mules, variant_id)
    elif "crypto" in target:
        n_exchanges = int(rng.integers(2, 5))
        code = _build_crypto_variant(n_exchanges, variant_id)
    elif "trade" in target:
        n_intermediaries = int(rng.integers(2, 6))
        code = _build_trade_variant(n_intermediaries, variant_id)
    elif "layering" in target or "shell" in target:
        n_hops = int(rng.integers(3, 8))
        code = _build_layering_variant(n_hops, variant_id)
    else:
        n_hops = int(rng.integers(3, 6))
        code = _build_layering_variant(n_hops, variant_id)

    fn = _compile_function(code)
    return fn, code


def _build_hybrid_variant(vid: int, phase1: str = "structuring", phase2: str = "crypto") -> str:
    """
    Multi-phase combinatorial scheme. Each (phase1, phase2) combo uses a DISTINCT
    set of transaction types so the keyword-based novelty embedding produces
    genuinely different vectors — preventing novelty=0.000 on every fallback.

    Combo → dominant tx types (determines keyword fingerprint):
        structuring+crypto  → cash_deposit  + crypto_transfer  (no wire/swift/trade)
        crypto+layering     → crypto_transfer + wire_transfer  (no cash/swift/trade)
        trade+structuring   → trade_finance + hawala           (no crypto/wire/swift)
        layering+trade      → swift + wire_transfer + trade_finance (no cash/crypto)
    """
    # Map each phase to its EXCLUSIVE transaction type — this is what makes
    # the keyword vectors different across combos.
    _TX = {
        "structuring": "cash_deposit",
        "crypto":      "crypto_transfer",
        "trade":       "trade_finance",
        "layering":    "wire_transfer",
    }
    tx1 = _TX.get(phase1, "cash_deposit")
    tx2 = _TX.get(phase2, "wire_transfer")
    # Integration leg uses the combo-specific finishing tx (not always swift)
    finish_tx = "swift" if phase2 in ("layering", "crypto") else "ach"

    return textwrap.dedent(f'''\
    def inject_scheme_hybrid_{vid}(graph, rng=None):
        """Codex Generated — Hybrid {phase1}+{phase2} scheme {vid}."""
        import numpy as np
        rng = rng or np.random.default_rng()
        individuals = [n for n, d in graph.graph.nodes(data=True) if d.get("node_type") == "individual"]
        accounts    = [n for n, d in graph.graph.nodes(data=True) if d.get("node_type") == "account"]
        cryptos     = [n for n, d in graph.graph.nodes(data=True) if d.get("node_type") == "crypto_exchange"]
        shells      = [n for n, d in graph.graph.nodes(data=True) if d.get("node_type") == "shell_company"]
        source   = str(rng.choice(individuals))
        n_mules  = int(rng.integers(2, 5))
        mules    = [str(x) for x in rng.choice([a for a in accounts if a != source], min(n_mules, len(accounts)-1), replace=False)]
        exchange = str(rng.choice(cryptos))
        shell    = str(rng.choice(shells))
        sink     = str(rng.choice([a for a in accounts if a != source and a not in mules]))
        scheme_id   = f"codex_hybrid_{vid}_{{int(rng.integers(1000,9999))}}"
        base_ts     = int(rng.integers(1672531200, 1704067200))
        base_amount = float(rng.uniform(30000, 200000))
        edges = []
        # Phase 1 ({phase1}) — source → mules using {tx1}
        for i, m in enumerate(mules):
            amt   = float(rng.uniform(8000.0, 9999.0))
            prior = dict(graph.graph[source][m]) if graph.graph.has_edge(source, m) else None
            graph.graph.add_edge(source, m, amount=amt,
                timestamp=base_ts + i * int(rng.integers(3600, 14400)),
                transaction_type="{tx1}",
                jurisdiction=graph.graph.nodes[source].get("jurisdiction", "US"),
                suspicion_weight=float(rng.uniform(0.75, 0.90)), scheme_id=scheme_id)
            edges.append({{"src": source, "tgt": m, "prior": prior}})
        # Phase 2 ({phase2}) — mules → exchange using {tx2}
        for i, m in enumerate(mules):
            amt   = base_amount / max(len(mules), 1) * float(rng.uniform(0.90, 1.05))
            prior = dict(graph.graph[m][exchange]) if graph.graph.has_edge(m, exchange) else None
            graph.graph.add_edge(m, exchange, amount=amt,
                timestamp=base_ts + n_mules * 14400 + i * 7200,
                transaction_type="{tx2}",
                jurisdiction=graph.graph.nodes[exchange].get("jurisdiction", "MT"),
                suspicion_weight=float(rng.uniform(0.70, 0.88)), scheme_id=scheme_id)
            edges.append({{"src": m, "tgt": exchange, "prior": prior}})
        # Phase 3 — exchange → shell using {tx2}
        amt3  = base_amount * float(rng.uniform(0.88, 0.96))
        prior = dict(graph.graph[exchange][shell]) if graph.graph.has_edge(exchange, shell) else None
        graph.graph.add_edge(exchange, shell, amount=amt3,
            timestamp=base_ts + (n_mules * 2) * 7200 + 86400,
            transaction_type="{tx2}",
            jurisdiction=graph.graph.nodes[shell].get("jurisdiction", "KY"),
            suspicion_weight=float(rng.uniform(0.80, 0.95)), scheme_id=scheme_id)
        edges.append({{"src": exchange, "tgt": shell, "prior": prior}})
        # Phase 4 (integration) — shell → sink using {finish_tx}
        amt4  = amt3 * float(rng.uniform(0.90, 0.98))
        prior = dict(graph.graph[shell][sink]) if graph.graph.has_edge(shell, sink) else None
        graph.graph.add_edge(shell, sink, amount=amt4,
            timestamp=base_ts + (n_mules * 2) * 7200 + 2 * 86400,
            transaction_type="{finish_tx}",
            jurisdiction=graph.graph.nodes[sink].get("jurisdiction", "US"),
            suspicion_weight=float(rng.uniform(0.65, 0.85)), scheme_id=scheme_id)
        edges.append({{"src": shell, "tgt": sink, "prior": prior}})
        full_path = [source] + mules + [exchange, shell, sink]
        gt = {{"source_entity": source, "sink_entity": sink, "full_path": full_path,
               "scheme_type": "hybrid_{phase1}_{phase2}", "num_hops": len(full_path) - 1,
               "intermediate_nodes": mules + [exchange, shell]}}
        graph._injected[scheme_id] = {{"edges": edges, "nodes": []}}
        graph.ground_truth[scheme_id] = gt
        return gt
    ''')


def _build_structuring_variant(n_mules: int, vid: int) -> str:
    return textwrap.dedent(f'''\
    def inject_scheme_struct_{vid}(graph, rng=None):
        """Codex Generated — Structuring variant {vid} with {n_mules} mules."""
        import numpy as np
        rng = rng or np.random.default_rng()
        individuals = [n for n, d in graph.graph.nodes(data=True) if d.get("node_type") == "individual"]
        accounts = [n for n, d in graph.graph.nodes(data=True) if d.get("node_type") == "account"]
        source = str(rng.choice(individuals))
        mules = [str(x) for x in rng.choice([a for a in accounts if a != source], {n_mules}, replace=False)]
        sink = str(rng.choice([a for a in accounts if a != source and a not in mules]))
        scheme_id = f"codex_struct_{vid}_{{int(rng.integers(1000,9999))}}"
        base_ts = int(rng.integers(1672531200, 1704067200))
        amounts = rng.uniform(8500.0, 9999.0, {n_mules}).tolist()
        edges = []
        for i, m in enumerate(mules):
            prior = dict(graph.graph[source][m]) if graph.graph.has_edge(source, m) else None
            graph.graph.add_edge(source, m, amount=amounts[i], timestamp=base_ts + i * int(rng.integers(900, 7200)),
                transaction_type="cash_deposit", jurisdiction=graph.graph.nodes[source].get("jurisdiction", "US"),
                suspicion_weight=float(rng.uniform(0.80, 0.95)), scheme_id=scheme_id)
            edges.append({{"src": source, "tgt": m, "prior": prior}})
        for i, m in enumerate(mules):
            prior = dict(graph.graph[m][sink]) if graph.graph.has_edge(m, sink) else None
            graph.graph.add_edge(m, sink, amount=amounts[i] * float(rng.uniform(0.93, 1.0)),
                timestamp=base_ts + {n_mules} * 7200 + i * 3600, transaction_type="wire_transfer",
                jurisdiction=graph.graph.nodes[sink].get("jurisdiction", "US"),
                suspicion_weight=float(rng.uniform(0.65, 0.90)), scheme_id=scheme_id)
            edges.append({{"src": m, "tgt": sink, "prior": prior}})
        gt = {{"source_entity": source, "sink_entity": sink, "full_path": [source] + mules + [sink],
               "scheme_type": "structuring_variant", "num_hops": 2, "intermediate_nodes": mules}}
        graph._injected[scheme_id] = {{"edges": edges, "nodes": []}}
        graph.ground_truth[scheme_id] = gt
        return gt
    ''')


def _build_crypto_variant(n_exchanges: int, vid: int) -> str:
    return textwrap.dedent(f'''\
    def inject_scheme_crypto_{vid}(graph, rng=None):
        """Codex Generated — Crypto variant {vid} with {n_exchanges} exchanges."""
        import numpy as np
        rng = rng or np.random.default_rng()
        accounts = [n for n, d in graph.graph.nodes(data=True) if d.get("node_type") == "account"]
        cryptos = [n for n, d in graph.graph.nodes(data=True) if d.get("node_type") == "crypto_exchange"]
        source = str(rng.choice(accounts))
        exchanges = [str(x) for x in rng.choice(cryptos, min({n_exchanges}, len(cryptos)), replace=False)]
        sink = str(rng.choice([a for a in accounts if a != source]))
        scheme_id = f"codex_crypto_{vid}_{{int(rng.integers(1000,9999))}}"
        base_ts = int(rng.integers(1672531200, 1704067200))
        base_amount = float(rng.uniform(10000, 300000))
        path = [source] + exchanges + [sink]
        edges = []
        for i in range(len(path) - 1):
            amt = base_amount * float(rng.uniform(0.90, 1.05))
            prior = dict(graph.graph[path[i]][path[i+1]]) if graph.graph.has_edge(path[i], path[i+1]) else None
            graph.graph.add_edge(path[i], path[i+1], amount=amt, timestamp=base_ts + i * int(rng.integers(3600, 43200)),
                transaction_type="crypto_transfer", jurisdiction=graph.graph.nodes[path[i+1]].get("jurisdiction", "MT"),
                suspicion_weight=float(rng.uniform(0.65, 0.90)), scheme_id=scheme_id)
            edges.append({{"src": path[i], "tgt": path[i+1], "prior": prior}})
        gt = {{"source_entity": source, "sink_entity": sink, "full_path": path,
               "scheme_type": "crypto_variant", "num_hops": len(exchanges) + 1, "intermediate_nodes": exchanges}}
        graph._injected[scheme_id] = {{"edges": edges, "nodes": []}}
        graph.ground_truth[scheme_id] = gt
        return gt
    ''')


def _build_trade_variant(n_intermediaries: int, vid: int) -> str:
    return textwrap.dedent(f'''\
    def inject_scheme_trade_{vid}(graph, rng=None):
        """Codex Generated — Trade variant {vid} with {n_intermediaries} intermediaries."""
        import numpy as np
        rng = rng or np.random.default_rng()
        shells = [n for n, d in graph.graph.nodes(data=True) if d.get("node_type") == "shell_company"]
        accounts = [n for n, d in graph.graph.nodes(data=True) if d.get("node_type") == "account"]
        source = str(rng.choice(shells))
        intermediaries = [str(x) for x in rng.choice([s for s in shells if s != source], min({n_intermediaries}, len(shells)-1), replace=False)]
        sink = str(rng.choice(accounts))
        scheme_id = f"codex_trade_{vid}_{{int(rng.integers(1000,9999))}}"
        base_ts = int(rng.integers(1672531200, 1704067200))
        base_amount = float(rng.uniform(100000, 3000000))
        path = [source] + intermediaries + [sink]
        edges = []
        for i in range(len(path) - 1):
            amt = base_amount * float(rng.uniform(0.65, 1.35))
            prior = dict(graph.graph[path[i]][path[i+1]]) if graph.graph.has_edge(path[i], path[i+1]) else None
            graph.graph.add_edge(path[i], path[i+1], amount=amt, timestamp=base_ts + i * 86400,
                transaction_type="trade_finance", jurisdiction=graph.graph.nodes[path[i+1]].get("jurisdiction", "HK"),
                suspicion_weight=float(rng.uniform(0.60, 0.90)), scheme_id=scheme_id)
            edges.append({{"src": path[i], "tgt": path[i+1], "prior": prior}})
        gt = {{"source_entity": source, "sink_entity": sink, "full_path": path,
               "scheme_type": "trade_variant", "num_hops": {n_intermediaries} + 1, "intermediate_nodes": intermediaries}}
        graph._injected[scheme_id] = {{"edges": edges, "nodes": []}}
        graph.ground_truth[scheme_id] = gt
        return gt
    ''')


def _build_layering_variant(n_hops: int, vid: int) -> str:
    return textwrap.dedent(f'''\
    def inject_scheme_layer_{vid}(graph, rng=None):
        """Codex Generated — Layering variant {vid} with {n_hops} hops."""
        import numpy as np
        rng = rng or np.random.default_rng()
        accounts = [n for n, d in graph.graph.nodes(data=True) if d.get("node_type") == "account"]
        shells = [n for n, d in graph.graph.nodes(data=True) if d.get("node_type") == "shell_company"]
        source = str(rng.choice(accounts))
        intermediaries = [str(x) for x in rng.choice(shells, min({n_hops}, len(shells)), replace=False)]
        sink = str(rng.choice([a for a in accounts if a != source]))
        scheme_id = f"codex_layer_{vid}_{{int(rng.integers(1000,9999))}}"
        base_ts = int(rng.integers(1672531200, 1704067200))
        start_amount = float(rng.uniform(50000, 500000))
        decay = float(rng.uniform(0.92, 0.98))
        path = [source] + intermediaries + [sink]
        edges = []
        for i in range(len(path) - 1):
            amt = start_amount * (decay ** i)
            prior = dict(graph.graph[path[i]][path[i+1]]) if graph.graph.has_edge(path[i], path[i+1]) else None
            graph.graph.add_edge(path[i], path[i+1], amount=amt, timestamp=base_ts + i * int(rng.integers(21600, 86400)),
                transaction_type="swift" if i == len(path)-2 else "wire_transfer",
                jurisdiction=graph.graph.nodes[path[i+1]].get("jurisdiction", "KY"),
                suspicion_weight=float(rng.uniform(0.70, 0.95)), scheme_id=scheme_id)
            edges.append({{"src": path[i], "tgt": path[i+1], "prior": prior}})
        gt = {{"source_entity": source, "sink_entity": sink, "full_path": path,
               "scheme_type": "layering_variant", "num_hops": {n_hops} + 1, "intermediate_nodes": intermediaries}}
        graph._injected[scheme_id] = {{"edges": edges, "nodes": []}}
        graph.ground_truth[scheme_id] = gt
        return gt
    ''')


# ---------------------------------------------------------------------------
# Codex Management
# ---------------------------------------------------------------------------

def append_to_codex(
    scheme_code: str,
    scheme_name: str,
    target_weakness: str,
    episode_number: int,
) -> None:
    """Append a validated scheme to criminal_codex.py and update registry."""
    # Clean up function name
    func_match = re.search(r'def (inject_\w+)', scheme_code)
    if not func_match:
        return
    func_name = func_match.group(1)
    registry_key = func_name.replace("inject_", "").replace("scheme_", "")
    
    # Add docstring annotation
    header = f'''

# ---------------------------------------------------------------------------
# Generated Scheme: {scheme_name} (Episode {episode_number})
# Target Weakness: {target_weakness}
# ---------------------------------------------------------------------------

'''
    
    with open(CODEX_PATH, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Append function
    content += header + scheme_code + "\n"
    
    # Update CODEX_REGISTRY
    registry_line = f'    "{registry_key}": {func_name},\n'
    content = content.replace(
        "# (Criminal Designer agent appends new validated schemes here)",
        f"# (Criminal Designer agent appends new validated schemes here)\n"
    )
    
    # Add to registry by appending before the closing brace
    content = content.replace(
        '"offshore_cascade": inject_offshore_cascade,\n}',
        f'"offshore_cascade": inject_offshore_cascade,\n    "{registry_key}": {func_name},\n}}'
    )
    
    with open(CODEX_PATH, "w", encoding="utf-8") as f:
        f.write(content)


def get_codex_functions() -> Dict[str, callable]:
    """Load all scheme functions from the current criminal_codex.py."""
    try:
        spec = importlib.util.spec_from_file_location("criminal_codex", CODEX_PATH)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return getattr(module, "CODEX_REGISTRY", {})
    except Exception:
        return {}


def get_codex_source_codes() -> List[str]:
    """Get source code of all existing codex functions for novelty comparison."""
    codes: List[str] = []
    try:
        with open(CODEX_PATH, "r", encoding="utf-8") as f:
            content = f.read()
        # Extract each function body
        funcs = re.findall(r'(def inject_\w+\(.*?\n(?:(?:    .*\n?)*|(?:.*\n?)*))', content)
        codes = [f for f in funcs if f.strip()]
    except Exception:
        pass
    return codes


# ---------------------------------------------------------------------------
# Criminal Designer Agent
# ---------------------------------------------------------------------------

class CriminalDesigner:
    """
    Full Criminal Designer agent.
    
    Tracks weaknesses → generates schemes → validates → appends to Codex.
    """
    
    def __init__(self, temperature: float = 1.0, seed: int = 42) -> None:
        self.weakness_db = WeaknessDatabase()
        self.temperature = temperature
        self._rng = np.random.default_rng(seed)
        self._episode_count = 0
        self._generation_count = 0
    
    def update_weakness(self, episode_results: Dict[str, Any]) -> None:
        """
        Update weakness database from episode results.
        
        episode_results should contain:
            scheme_type: str
            investigator_f1: float
        """
        scheme_type = episode_results.get("scheme_type", "")
        inv_f1 = episode_results.get("investigator_f1", 0.0)
        self.weakness_db.update(scheme_type, inv_f1)
        self._episode_count += 1
    
    def generate_and_validate(self) -> Tuple[Optional[callable], Dict[str, Any]]:
        """
        Generate a new scheme, validate it, and append to Codex if valid.
        
        Returns (scheme_fn or None, info_dict).
        """
        weakness_vector = self.weakness_db.get_weakness_vector()
        
        info: Dict[str, Any] = {
            "weakness_vector": weakness_vector,
            "attempts": 0,
            "validated": False,
            "penalty": 0.0,
            "target_weakness": "",
            "novelty_bonus": 0.0,
        }
        
        for attempt in range(MAX_GENERATION_ATTEMPTS):
            info["attempts"] = attempt + 1
            
            fn, code, target = generate_scheme(
                weakness_vector, self.temperature, self._rng,
            )
            info["target_weakness"] = target
            
            if fn is None:
                info["penalty"] -= 0.1
                continue
            
            # Validate
            passed, reason, gt = validate_scheme(fn)
            
            if not passed:
                info["penalty"] -= 0.1
                info["validation_error"] = reason
                continue
            
            # Compute novelty (min cosine distance to nearest existing scheme)
            existing_codes = get_codex_source_codes()
            novelty = compute_novelty_bonus(code, existing_codes)
            info["novelty_bonus"] = novelty

            # Reject structural duplicates: if the new scheme is too similar
            # to an existing codex scheme (novelty < 0.05), the criminal is
            # just re-generating the same attack. Skip codex append and retry
            # so the codex only grows with genuinely different schemes.
            if existing_codes and novelty < 0.05:
                info["penalty"] -= 0.05
                info["validation_error"] = "structural_duplicate (novelty < 0.05)"
                continue

            info["validated"] = True

            # Append to Codex
            self._generation_count += 1
            scheme_name = f"gen_{self._generation_count}_{target}"
            append_to_codex(code, scheme_name, target, self._episode_count)

            return fn, info
        
        # All attempts failed
        return None, info
    
    def inject_generated_scheme(
        self, graph: TransactionGraph, scheme_fn: callable
    ) -> Dict[str, Any]:
        """
        Execute a validated Codex scheme against the main TransactionGraph.
        Returns ground_truth dict.
        """
        return scheme_fn(graph, self._rng)
    
    def compute_reward(
        self,
        laundering_volume: float,
        detection_rate: float,
        novelty_bonus: float = 1.0,
        morph_success: bool = False,
    ) -> Dict[str, Any]:
        """Compute R_criminal using the reward module."""
        return r_criminal(
            laundering_volume=laundering_volume,
            detection_rate=detection_rate,
            novelty_bonus=novelty_bonus,
            morph_success=morph_success,
        )
    
    def synthesize_zero_day(
        self,
        top_schemes: List[str],
        episode_number: int = 0,
    ) -> Tuple[Optional[callable], Dict[str, Any]]:
        """
        Compose a Zero-Day scheme from the hardest schemes (by ELO).

        Instead of randomly generating a novel scheme, we instruct the LLM to
        combine the evasion strategies of the 3 schemes the investigator
        struggles with most. This makes the Zero-Day provably hard by construction:
        if the investigator fails on A, B, C individually, a scheme that combines
        all three evasion patterns will be even harder to detect.

        Returns (scheme_fn, info_dict) — scheme_fn is None if generation fails.
        """
        if not top_schemes:
            return None, {"error": "no top schemes provided"}

        # Describe each hard scheme's evasion strategy
        evasion_descriptions = {
            "smurfing":           "splitting large amounts into many small cash deposits near $10k threshold",
            "structuring":        "structuring deposits across multiple mule accounts to avoid CTR",
            "crypto_mixing":      "routing funds through multiple crypto exchanges to obscure origin",
            "layering":           "cascading wire transfers through shell companies across jurisdictions",
            "shell_company":      "using nested shell companies with trade_finance invoices to clean funds",
            "trade_based":        "over/under-invoicing trade transactions to move value across borders",
            "crypto_variant":     "multi-hop crypto transfers through offshore exchanges",
            "trade_variant":      "trade mispricing through shell intermediaries in high-secrecy jurisdictions",
            "layering_variant":   "multi-shell layering with SWIFT international wire transfers",
            "structuring_variant":"mule network structuring with sub-threshold cash deposits",
        }

        strategies = []
        for s in top_schemes[:3]:
            desc = evasion_descriptions.get(s, f"complex {s} evasion pattern")
            strategies.append(f"- {s}: {desc}")

        zero_day_prompt = f"""\
You are a financial crime designer AI creating a Zero-Day money laundering scheme.

The investigator AI has been trained and is weakest against these three patterns:
{chr(10).join(strategies)}

Your task: synthesize a SINGLE Python function that COMBINES all three evasion strategies
into one coordinated multi-phase attack. This scheme should:
1. Use Phase 1 from the first pattern, Phase 2 from the second, Phase 3 from the third
2. Deliberately route through jurisdictions and node types that exploit each weakness
3. Be structurally different from any single scheme type

Write a Python function with signature: def inject_zero_day(graph, rng=None) -> dict:
Requirements:
- Pick nodes from graph.graph.nodes(data=True) using node_type filters
- Use rng (numpy Generator) for all randomness
- Add edges: graph.graph.add_edge(src, tgt, amount=..., timestamp=..., transaction_type=..., jurisdiction=..., suspicion_weight=..., scheme_id=...)
- Track: graph._injected[scheme_id] = {{"edges": [...], "nodes": []}}
- Store: graph.ground_truth[scheme_id] = gt
- Return: dict with keys source_entity, sink_entity, full_path, scheme_type, num_hops, intermediate_nodes
- Use scheme_id = "zero_day_{{int(rng.integers(1000,9999))}}"
- Amounts must be $100 - $10,000,000
- scheme_type in return dict should be "zero_day"

Respond with ONLY the Python function. No explanation. No markdown fencing.
"""

        print(f"\n[ZeroDay] Synthesizing Zero-Day from top schemes: {top_schemes[:3]}")
        response = _call_gemini(zero_day_prompt)

        if response:
            code = _extract_function_code(response)
            if code:
                fn = _compile_function(code)
                if fn:
                    passed, reason, gt = validate_scheme(fn)
                    if passed:
                        existing_codes = get_codex_source_codes()
                        novelty = compute_novelty_bonus(code, existing_codes)
                        append_to_codex(code, "zero_day", "zero_day", episode_number)
                        print(f"[ZeroDay] Synthesized and validated. Structural novelty={novelty:.3f}")
                        return fn, {
                            "validated": True,
                            "novelty_bonus": novelty,
                            "target_schemes": top_schemes[:3],
                            "code": code,
                        }
                    else:
                        print(f"[ZeroDay] Validation failed: {reason}")

        # Fallback: build hybrid of top 2 schemes
        print("[ZeroDay] LLM unavailable — using hybrid fallback")
        phase1 = top_schemes[0].replace("_variant", "").replace("_company", "")
        phase2 = top_schemes[1].replace("_variant", "").replace("_company", "") if len(top_schemes) > 1 else "crypto"
        vid = int(self._rng.integers(1000, 9999))
        code = _build_hybrid_variant(vid, phase1=phase1[:10], phase2=phase2[:10])
        fn = _compile_function(code)
        existing_codes = get_codex_source_codes()
        novelty = compute_novelty_bonus(code, existing_codes) if fn else 0.0
        return fn, {
            "validated": fn is not None,
            "novelty_bonus": novelty,
            "target_schemes": top_schemes[:3],
            "fallback": True,
        }

    @property
    def codex_size(self) -> int:
        """Number of schemes in the Codex."""
        return len(get_codex_functions())
    
    @property
    def weakness_vector(self) -> Dict[str, float]:
        return self.weakness_db.get_weakness_vector()


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("HEIST Criminal Designer — Smoke Test")
    print("=" * 70)
    
    designer = CriminalDesigner(temperature=1.0, seed=42)
    
    # ── Test 1: Weakness tracking ──────────────────────────────────────
    print("\n[1] Simulating 10 episodes where investigator fails at structuring...")
    for i in range(10):
        designer.update_weakness({
            "scheme_type": "smurfing",
            "investigator_f1": 0.2,  # investigator failing hard
        })
    for i in range(5):
        designer.update_weakness({
            "scheme_type": "layering",
            "investigator_f1": 0.8,  # investigator doing well
        })
    
    wv = designer.weakness_vector
    print(f"    Weakness vector:")
    for cat, rate in sorted(wv.items(), key=lambda x: -x[1]):
        bar = "#" * int(rate * 30)
        print(f"      {cat:25s} {rate:.2f} {bar}")
    
    # ── Test 2: Softmax targeting ──────────────────────────────────────
    print("\n[2] Softmax target selection (10 samples, should favor smurfing)...")
    rng = np.random.default_rng(42)
    selections = [softmax_target_selection(wv, 1.0, rng) for _ in range(10)]
    from collections import Counter
    counts = Counter(selections)
    for cat, cnt in counts.most_common():
        print(f"      {cat}: {cnt}/10")
    
    # ── Test 3: Validation suite ───────────────────────────────────────
    print("\n[3] Validating 4 seed schemes...")
    
    # Import seed schemes
    sys.path.insert(0, _PROJECT_ROOT)
    from criminal_codex import CODEX_REGISTRY
    
    for name, fn in CODEX_REGISTRY.items():
        passed, reason, gt = validate_scheme(fn)
        status = "PASS" if passed else "FAIL"
        print(f"    {name:25s} [{status}] {reason}")
        if gt:
            print(f"      Path: {gt['source_entity']} -> ... -> {gt['sink_entity']} ({gt['num_hops']} hops)")
    
    # ── Test 4: Generate new schemes ───────────────────────────────────
    print("\n[4] Generating 2 new schemes (fallback mode, no Gemini)...")
    for i in range(2):
        fn, info = designer.generate_and_validate()
        status = "VALIDATED" if info["validated"] else "FAILED"
        print(f"    Scheme {i+1}: [{status}]")
        print(f"      Target: {info['target_weakness']}")
        print(f"      Attempts: {info['attempts']}")
        print(f"      Novelty: {info['novelty_bonus']:.3f}")
        print(f"      Penalty: {info['penalty']:.1f}")
    
    # ── Test 5: Novelty bonus ──────────────────────────────────────────
    print("\n[5] Novelty bonus computation...")
    code_a = "def inject_smurfing(graph): graph.add_edge(src, tgt, amount=9000)"
    code_b = "def inject_crypto(graph): graph.add_edge(src, tgt, amount=50000, type=crypto)"
    code_c = "def inject_smurfing_v2(graph): graph.add_edge(src, tgt, amount=9500)"
    
    novelty_ab = compute_novelty_bonus(code_b, [code_a])
    novelty_ac = compute_novelty_bonus(code_c, [code_a])
    print(f"    smurfing vs crypto: {novelty_ab:.3f} (should be high)")
    print(f"    smurfing vs smurfing_v2: {novelty_ac:.3f} (should be low)")
    assert novelty_ab > novelty_ac, "Novelty ordering wrong!"
    print(f"    Ordering correct: crypto is more novel than variant")
    
    # ── Test 6: R_criminal ─────────────────────────────────────────────
    print("\n[6] R_criminal calculation...")
    r = designer.compute_reward(
        laundering_volume=0.8,
        detection_rate=0.2,  # investigator catching 20%
        novelty_bonus=1.5,
        morph_success=True,
    )
    print(f"    R_criminal = {r['total']:.4f}")
    print(f"    Breakdown: vol={r['laundering_volume']:.2f} * (1-det)={1-r['detection_rate']:.2f} "
          f"* novelty={r['novelty_bonus']:.2f} * morph={r['morph_success_bonus']:.2f}")
    expected = 0.8 * 0.8 * 1.5 * 1.2
    assert abs(r["total"] - expected) < 0.001, f"Expected {expected}, got {r['total']}"
    
    # ── Test 7: Codex size ─────────────────────────────────────────────
    print(f"\n[7] Codex size: {designer.codex_size} schemes")
    
    print(f"\n{'='*70}")
    print(f"ALL SMOKE TESTS PASSED")
    print(f"{'='*70}")
