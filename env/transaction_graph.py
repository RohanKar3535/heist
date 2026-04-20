"""
TransactionGraph — 100k nodes, 1M+ edges, IBM AMLSim statistical properties.
Supports scheme injection, clean removal, and sub-500ms reset.
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any

import networkx as nx
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

JURISDICTIONS: list[str] = [
    "US", "UK", "DE", "FR", "SG", "HK", "CH", "JP", "AU", "CA",
    "NL", "LU", "IE", "CY", "MT", "BS", "KY", "VG", "BZ", "PA",
]
OFFSHORE_JURISDICTIONS: list[str] = [
    "KY", "BS", "VG", "BZ", "PA", "CH", "LU", "CY", "MT", "IE",
]
CRYPTO_JURISDICTIONS: list[str] = [
    "MT", "SG", "HK", "US", "UK", "EE", "LT",
]
TRANSACTION_TYPES: list[str] = [
    "wire_transfer", "cash_deposit", "crypto_transfer",
    "check", "ach", "swift", "hawala", "trade_finance",
]
SCHEME_TYPES: list[str] = [
    "smurfing", "layering", "shell_company", "crypto_mixing", "trade_based",
]

_BASE_TS = int(datetime(2023, 1, 1).timestamp())
_END_TS  = int(datetime(2025, 1, 1).timestamp())


# ---------------------------------------------------------------------------
# TransactionGraph
# ---------------------------------------------------------------------------

class TransactionGraph:
    """
    Directed transaction network with IBM AMLSim statistical properties.

    Node types  : account (60k), shell_company (20k), individual (15k), crypto_exchange (5k)
    Edge attrs  : amount, timestamp, transaction_type, jurisdiction, suspicion_weight, scheme_id
    Degree dist : power-law via Pareto(α=2) source weights + uniform target sampling
    Normal txs  : amount ~ Gamma(2, 5000)
    Structuring : amount ~ Uniform(9000, 9999)  [injected by smurfing scheme]
    """

    def __init__(self, seed: int = 42) -> None:
        self._rng                              = np.random.default_rng(seed)
        self._nodes_by_type: dict[str, list[str]] = {}
        self._injected: dict[str, dict]        = {}   # scheme_id → {edges, nodes}
        self.ground_truth: dict[str, dict]     = {}   # scheme_id → ground truth record
        self.graph: nx.DiGraph                 = nx.DiGraph()
        self._build()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _build(self) -> None:
        G   = self.graph
        rng = self._rng

        # ── Nodes ──────────────────────────────────────────────────────
        node_specs = [
            ("acc",    60_000, "account",         JURISDICTIONS),
            ("shell",  20_000, "shell_company",    OFFSHORE_JURISDICTIONS),
            ("ind",    15_000, "individual",        JURISDICTIONS),
            ("crypto",  5_000, "crypto_exchange",   CRYPTO_JURISDICTIONS),
        ]

        all_ids: list[str] = []
        for prefix, count, node_type, jur_pool in node_specs:
            jurs = rng.choice(jur_pool, count).tolist()
            ids  = [f"{prefix}_{i}" for i in range(count)]
            self._nodes_by_type[node_type] = ids
            G.add_nodes_from(
                (nid, {"node_type": node_type, "jurisdiction": j})
                for nid, j in zip(ids, jurs)
            )
            all_ids.extend(ids)

        # ── Edges ──────────────────────────────────────────────────────
        n      = len(all_ids)          # 100 000
        id_arr = np.array(all_ids)

        # Pareto α=2 source weights → power-law out-degree (IBM AMLSim property)
        raw     = rng.pareto(2.0, n) + 1.0
        weights = raw / raw.sum()

        # Attempt 1.3M to ensure ≥1M survive self-loop removal + dedup in DiGraph
        attempt = 1_300_000
        src_idx = rng.choice(n, size=attempt, p=weights)
        tgt_idx = rng.integers(0, n, size=attempt)
        mask    = src_idx != tgt_idx
        src_idx, tgt_idx = src_idx[mask], tgt_idx[mask]

        m = len(src_idx)

        # IBM AMLSim: normal transactions ~ Gamma(shape=2, scale=5000), mean ≈ $10k
        amounts = rng.gamma(2, 5_000, m)
        ts      = rng.integers(_BASE_TS, _END_TS, m)
        tt_idx  = rng.integers(0, len(TRANSACTION_TYPES), m)
        jur_idx = rng.integers(0, len(JURISDICTIONS), m)
        sw      = rng.beta(1, 9, m)   # suspicion_weight: mostly near 0, few outliers

        tt_arr  = np.array(TRANSACTION_TYPES)
        jur_arr = np.array(JURISDICTIONS)

        G.add_edges_from(
            (
                str(id_arr[src_idx[i]]),
                str(id_arr[tgt_idx[i]]),
                {
                    "amount":           float(amounts[i]),
                    "timestamp":        int(ts[i]),
                    "transaction_type": str(tt_arr[tt_idx[i]]),
                    "jurisdiction":     str(jur_arr[jur_idx[i]]),
                    "suspicion_weight": float(sw[i]),
                    "scheme_id":        None,
                },
            )
            for i in range(m)
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _pick(self, node_type: str, k: int = 1, exclude: set[str] | None = None) -> list[str]:
        """Sample k unique node IDs of given type, avoiding excluded set."""
        pool = self._nodes_by_type[node_type]
        if exclude:
            pool = [x for x in pool if x not in exclude]
        k = min(k, len(pool))
        return [str(x) for x in self._rng.choice(pool, k, replace=False).tolist()]

    def _chain_timestamps(self, n_steps: int) -> list[int]:
        """Return n_steps monotonically increasing timestamps spaced 1h–24h apart."""
        base = int(self._rng.integers(_BASE_TS, _END_TS - 86_400 * n_steps))
        gap  = int(self._rng.integers(3_600, 86_400))
        return [base + i * gap for i in range(n_steps)]

    def _add_edge(self, scheme_id: str, src: str, tgt: str, attrs: dict) -> dict:
        """Add a scheme edge, preserving any prior background edge for restoration."""
        prior = dict(self.graph[src][tgt]) if self.graph.has_edge(src, tgt) else None
        self.graph.add_edge(src, tgt, **attrs)
        return {"src": src, "tgt": tgt, "prior": prior}

    def _register(
        self,
        scheme_id: str,
        edges: list[dict],
        nodes: list[str],
        gt: dict,
    ) -> str:
        self._injected[scheme_id]  = {"edges": edges, "nodes": nodes}
        self.ground_truth[scheme_id] = gt
        return scheme_id

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def inject_scheme(self, scheme_type: str, params: dict | None = None) -> str:
        """
        Plant a laundering pattern into the graph.

        Args:
            scheme_type: one of SCHEME_TYPES
            params:      optional overrides (e.g. n_hops, n_smurfs)

        Returns:
            scheme_id string for later clear_scheme() calls
        """
        if scheme_type not in SCHEME_TYPES:
            raise ValueError(f"Unknown scheme_type '{scheme_type}'. Valid: {SCHEME_TYPES}")
        params    = params or {}
        scheme_id = f"scheme_{len(self._injected)}_{scheme_type}_{int(self._rng.integers(1000, 9999))}"
        dispatch  = {
            "smurfing":      self._inject_smurfing,
            "layering":      self._inject_layering,
            "shell_company": self._inject_shell_company,
            "crypto_mixing": self._inject_crypto_mixing,
            "trade_based":   self._inject_trade_based,
        }
        return dispatch[scheme_type](scheme_id, params)

    def clear_scheme(self, scheme_id: str) -> None:
        """
        Remove all edges (and any new nodes) added by the given scheme.
        Restores any background edges that were overwritten.
        """
        if scheme_id not in self._injected:
            raise KeyError(f"Scheme '{scheme_id}' not found.")
        info = self._injected.pop(scheme_id)
        self.ground_truth.pop(scheme_id, None)

        # Remove scheme nodes first (cascades their edges automatically)
        for node in info["nodes"]:
            if self.graph.has_node(node):
                self.graph.remove_node(node)

        # Restore / remove scheme edges
        for rec in info["edges"]:
            src, tgt = rec["src"], rec["tgt"]
            if self.graph.has_edge(src, tgt):
                self.graph.remove_edge(src, tgt)
            if rec["prior"] is not None:
                self.graph.add_edge(src, tgt, **rec["prior"])

    def reset(self) -> None:
        """Remove all injected schemes, restoring graph to clean baseline. Target: <500ms."""
        for sid in list(self._injected.keys()):
            self.clear_scheme(sid)

    def get_neighbors(self, entity_id: str) -> dict[str, Any]:
        """
        Return direct connections only (partial observability for the investigator).

        Returns dict with keys: entity_id, node_type, jurisdiction, outgoing, incoming.
        Each entry in outgoing/incoming carries all edge attributes.
        """
        if not self.graph.has_node(entity_id):
            return {"error": f"Entity '{entity_id}' not found."}

        node_data = dict(self.graph.nodes[entity_id])

        outgoing = [
            {"entity_id": tgt, **dict(self.graph.nodes[tgt]), **dict(data)}
            for tgt, data in self.graph[entity_id].items()
        ]
        incoming = [
            {"entity_id": src, **dict(self.graph.nodes[src]), **dict(self.graph[src][entity_id])}
            for src in self.graph.predecessors(entity_id)
        ]

        return {"entity_id": entity_id, **node_data, "outgoing": outgoing, "incoming": incoming}

    # ------------------------------------------------------------------
    # Scheme injectors
    # ------------------------------------------------------------------

    def _inject_smurfing(self, scheme_id: str, params: dict) -> str:
        """
        Smurfing / structuring: multiple deposits just below $10k reporting threshold.
        Pattern: individual → N smurf accounts → aggregation account.
        Amounts: Uniform(9000, 9999) per IBM AMLSim structuring distribution.
        """
        rng      = self._rng
        n_smurfs = int(params.get("n_smurfs", rng.integers(3, 11)))
        source   = self._pick("individual")[0]
        smurfs   = self._pick("account", n_smurfs, exclude={source})
        sink     = self._pick("account", 1, exclude={source} | set(smurfs))[0]

        ts      = self._chain_timestamps(n_smurfs + 1)
        amounts = rng.uniform(9_000.0, 9_999.0, n_smurfs).tolist()

        edges: list[dict] = []
        # Placement: source → each smurf (structuring deposits)
        for i, smurf in enumerate(smurfs):
            edges.append(self._add_edge(scheme_id, source, smurf, {
                "amount":           amounts[i],
                "timestamp":        ts[i],
                "transaction_type": "cash_deposit",
                "jurisdiction":     self.graph.nodes[source]["jurisdiction"],
                "suspicion_weight": float(rng.uniform(0.80, 0.95)),
                "scheme_id":        scheme_id,
            }))
        # Aggregation: each smurf → sink
        for i, smurf in enumerate(smurfs):
            edges.append(self._add_edge(scheme_id, smurf, sink, {
                "amount":           amounts[i] * float(rng.uniform(0.95, 1.00)),
                "timestamp":        ts[n_smurfs],
                "transaction_type": "wire_transfer",
                "jurisdiction":     self.graph.nodes[sink]["jurisdiction"],
                "suspicion_weight": float(rng.uniform(0.70, 0.90)),
                "scheme_id":        scheme_id,
            }))

        return self._register(scheme_id, edges, [], {
            "source_entity":    source,
            "sink_entity":      sink,
            "full_path":        [source] + smurfs + [sink],
            "scheme_type":      "smurfing",
            "num_hops":         2,
            "intermediate_nodes": smurfs,
        })

    def _inject_layering(self, scheme_id: str, params: dict) -> str:
        """
        Layering: funds routed through 3–8 shell company hops across jurisdictions.
        Amount decays slightly each hop (transaction fees).
        """
        rng    = self._rng
        n_hops = int(params.get("n_hops", rng.integers(3, 9)))
        source = self._pick("account")[0]
        shells = self._pick("shell_company", n_hops)
        sink   = self._pick("account", 1, exclude={source})[0]

        path         = [source] + shells + [sink]
        ts           = self._chain_timestamps(len(path))
        start_amount = float(rng.uniform(50_000, 500_000))
        decay        = float(rng.uniform(0.970, 0.998))

        edges: list[dict] = []
        for i in range(len(path) - 1):
            edges.append(self._add_edge(scheme_id, path[i], path[i + 1], {
                "amount":           start_amount * (decay ** i),
                "timestamp":        ts[i],
                "transaction_type": "swift" if i == len(path) - 2 else "wire_transfer",
                "jurisdiction":     self.graph.nodes[path[i + 1]]["jurisdiction"],
                "suspicion_weight": float(rng.uniform(0.75, 0.95)),
                "scheme_id":        scheme_id,
            }))

        return self._register(scheme_id, edges, [], {
            "source_entity":    source,
            "sink_entity":      sink,
            "full_path":        path,
            "scheme_type":      "layering",
            "num_hops":         n_hops + 1,
            "intermediate_nodes": shells,
        })

    def _inject_shell_company(self, scheme_id: str, params: dict) -> str:
        """
        Shell company network: individual moves funds through 2–4 offshore shells.
        Classic placement → layering → integration via corporate structures.
        """
        rng      = self._rng
        n_shells = int(params.get("n_shells", rng.integers(2, 5)))
        source   = self._pick("individual")[0]
        shells   = self._pick("shell_company", n_shells)
        sink     = self._pick("account")[0]

        path        = [source] + shells + [sink]
        ts          = self._chain_timestamps(len(path))
        base_amount = float(rng.uniform(100_000, 2_000_000))

        edges: list[dict] = []
        for i in range(len(path) - 1):
            edges.append(self._add_edge(scheme_id, path[i], path[i + 1], {
                "amount":           base_amount * float(rng.uniform(0.90, 1.00)),
                "timestamp":        ts[i],
                "transaction_type": "wire_transfer",
                "jurisdiction":     self.graph.nodes[path[i + 1]]["jurisdiction"],
                "suspicion_weight": float(rng.uniform(0.80, 0.95)),
                "scheme_id":        scheme_id,
            }))

        return self._register(scheme_id, edges, [], {
            "source_entity":    source,
            "sink_entity":      sink,
            "full_path":        path,
            "scheme_type":      "shell_company",
            "num_hops":         n_shells + 1,
            "intermediate_nodes": shells,
        })

    def _inject_crypto_mixing(self, scheme_id: str, params: dict) -> str:
        """
        Crypto mixing: account → chain of crypto exchanges → clean account.
        Obfuscates trail via multiple exchange hops.
        """
        rng    = self._rng
        n_hops = int(params.get("n_hops", rng.integers(2, 6)))
        source = self._pick("account")[0]
        cryptos = self._pick(
            "crypto_exchange",
            min(n_hops, len(self._nodes_by_type["crypto_exchange"])),
        )
        sink = self._pick("account", 1, exclude={source})[0]

        path        = [source] + cryptos + [sink]
        ts          = self._chain_timestamps(len(path))
        base_amount = float(rng.uniform(10_000, 200_000))

        edges: list[dict] = []
        for i in range(len(path) - 1):
            edges.append(self._add_edge(scheme_id, path[i], path[i + 1], {
                "amount":           base_amount * float(rng.uniform(0.92, 1.00)),
                "timestamp":        ts[i],
                "transaction_type": "crypto_transfer",
                "jurisdiction":     self.graph.nodes[path[i + 1]]["jurisdiction"],
                "suspicion_weight": float(rng.uniform(0.70, 0.95)),
                "scheme_id":        scheme_id,
            }))

        return self._register(scheme_id, edges, [], {
            "source_entity":    source,
            "sink_entity":      sink,
            "full_path":        path,
            "scheme_type":      "crypto_mixing",
            "num_hops":         len(cryptos) + 1,
            "intermediate_nodes": cryptos,
        })

    def _inject_trade_based(self, scheme_id: str, params: dict) -> str:
        """
        Trade-based laundering: over/under-invoicing across shell companies.
        Large amounts with ±30% invoice variance to obscure true value.
        """
        rng   = self._rng
        n_mid = int(params.get("n_intermediaries", rng.integers(2, 5)))
        source = self._pick("shell_company")[0]

        n_sh = max(1, n_mid // 2)
        n_ac = n_mid - n_sh
        intermediaries: list[str] = (
            self._pick("shell_company", n_sh, exclude={source})
            + self._pick("account", n_ac)
        )
        rng.shuffle(intermediaries)
        sink = self._pick("account", 1, exclude=set(intermediaries))[0]

        path        = [source] + intermediaries + [sink]
        ts          = self._chain_timestamps(len(path))
        base_amount = float(rng.uniform(500_000, 5_000_000))

        edges: list[dict] = []
        for i in range(len(path) - 1):
            edges.append(self._add_edge(scheme_id, path[i], path[i + 1], {
                "amount":           base_amount * float(rng.uniform(0.70, 1.30)),
                "timestamp":        ts[i],
                "transaction_type": "trade_finance",
                "jurisdiction":     self.graph.nodes[path[i + 1]]["jurisdiction"],
                "suspicion_weight": float(rng.uniform(0.65, 0.95)),
                "scheme_id":        scheme_id,
            }))

        return self._register(scheme_id, edges, [], {
            "source_entity":    source,
            "sink_entity":      sink,
            "full_path":        path,
            "scheme_type":      "trade_based",
            "num_hops":         n_mid + 1,
            "intermediate_nodes": intermediaries,
        })


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    print("=" * 60)
    print("HEIST TransactionGraph — Smoke Tests")
    print("=" * 60)

    # ── Build ──────────────────────────────────────────────────────────
    print("\n[1] Building graph (100k nodes, 1M+ edges)...")
    t0 = time.time()
    tg = TransactionGraph(seed=42)
    build_time = time.time() - t0
    n_nodes = tg.graph.number_of_nodes()
    n_edges = tg.graph.number_of_edges()
    print(f"    Build time : {build_time:.1f}s")
    print(f"    Nodes      : {n_nodes:,}")
    print(f"    Edges      : {n_edges:,}")

    expected_types = {
        "account": 60_000, "shell_company": 20_000,
        "individual": 15_000, "crypto_exchange": 5_000,
    }
    type_counts: dict[str, int] = {}
    for _, d in tg.graph.nodes(data=True):
        nt = d["node_type"]
        type_counts[nt] = type_counts.get(nt, 0) + 1
    for nt, expected in expected_types.items():
        assert type_counts[nt] == expected, f"Node count mismatch: {nt} = {type_counts[nt]}"
    print(f"    Node types : {type_counts}")
    assert n_nodes == 100_000, f"Expected 100k nodes, got {n_nodes}"
    assert n_edges >= 1_000_000, f"Expected ≥1M edges, got {n_edges}"
    print("    PASS: node counts and edge count")

    # ── Inject 5 schemes ──────────────────────────────────────────────
    print("\n[2] Injecting 5 scheme types...")
    scheme_ids: list[str] = []
    for stype in SCHEME_TYPES:
        sid = tg.inject_scheme(stype)
        scheme_ids.append(sid)
        gt  = tg.ground_truth[sid]
        print(f"    {stype:15s} → source={gt['source_entity']}, "
              f"sink={gt['sink_entity']}, hops={gt['num_hops']}, "
              f"path_len={len(gt['full_path'])}")

        # Verify every recorded scheme edge actually exists in the graph
        for rec in tg._injected[sid]["edges"]:
            src, tgt = rec["src"], rec["tgt"]
            assert tg.graph.has_edge(src, tgt), \
                f"Missing edge {src} → {tgt} for scheme {sid}"
        # Verify ground truth nodes are present
        assert tg.graph.has_node(gt["source_entity"])
        assert tg.graph.has_node(gt["sink_entity"])
        for node in gt["intermediate_nodes"]:
            assert tg.graph.has_node(node)

    print("    PASS: all ground truth paths verified")

    # ── clear_scheme (single) ─────────────────────────────────────────
    print("\n[3] Testing clear_scheme() on smurfing scheme...")
    smurfing_sid  = scheme_ids[0]
    smurfing_edges = [(r["src"], r["tgt"]) for r in tg._injected[smurfing_sid]["edges"]]
    tg.clear_scheme(smurfing_sid)
    assert smurfing_sid not in tg.ground_truth
    assert smurfing_sid not in tg._injected
    for src, tgt in smurfing_edges:
        # Edge should be gone (or restored to prior non-scheme data)
        if tg.graph.has_edge(src, tgt):
            assert tg.graph[src][tgt].get("scheme_id") != smurfing_sid, \
                f"Scheme edge still present: {src} → {tgt}"
    print("    PASS: clear_scheme removes scheme edges")

    # ── reset() ───────────────────────────────────────────────────────
    print("\n[4] Testing reset() timing and completeness...")
    # Re-inject all 5 to ensure there's something to reset
    for stype in SCHEME_TYPES:
        tg.inject_scheme(stype)
    edges_before = tg.graph.number_of_edges()

    t0 = time.time()
    tg.reset()
    reset_ms = (time.time() - t0) * 1_000

    print(f"    Reset time : {reset_ms:.1f}ms")
    assert reset_ms < 500, f"reset() too slow: {reset_ms:.1f}ms (limit 500ms)"
    assert len(tg._injected)   == 0, "Schemes still registered after reset"
    assert len(tg.ground_truth) == 0, "Ground truth not cleared after reset"
    # No edge should carry a scheme_id
    scheme_edges = [
        (u, v) for u, v, d in tg.graph.edges(data=True)
        if d.get("scheme_id") is not None
    ]
    assert len(scheme_edges) == 0, f"{len(scheme_edges)} scheme edges remain after reset"
    print("    PASS: reset() cleans completely and is fast")

    # ── get_neighbors() ───────────────────────────────────────────────
    print("\n[5] Testing get_neighbors() (partial observability)...")
    sample = "acc_100"
    nb = tg.get_neighbors(sample)
    assert nb["entity_id"]  == sample
    assert nb["node_type"]  == "account"
    assert "jurisdiction"   in nb
    assert isinstance(nb["outgoing"], list)
    assert isinstance(nb["incoming"], list)
    print(f"    {sample}: {len(nb['outgoing'])} outgoing, {len(nb['incoming'])} incoming")
    # Verify each neighbour entry has required edge attributes
    for entry in nb["outgoing"][:3]:
        for key in ("entity_id", "node_type", "amount", "timestamp",
                    "transaction_type", "jurisdiction", "suspicion_weight"):
            assert key in entry, f"Missing key '{key}' in outgoing entry"
    # Test unknown entity
    bad = tg.get_neighbors("nonexistent_999")
    assert "error" in bad
    print("    PASS: get_neighbors returns correct structure")

    print("\n" + "=" * 60)
    print("ALL SMOKE TESTS PASSED")
    print("=" * 60)
