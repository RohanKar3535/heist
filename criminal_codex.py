"""
Criminal Codex - Seed Schemes.

This file starts with 4 seed scheme generators and grows as the Criminal
Designer agent generates new validated schemes during training.

Each scheme function has the signature:
    def inject_scheme_XXX(graph, rng) -> dict
        Returns ground_truth dict with: source_entity, sink_entity, full_path,
        scheme_type, num_hops, intermediate_nodes

All schemes are validated before entry (see agents/criminal.py validation suite).
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Seed Scheme 1: Rapid Structuring (smurfing variant)
# ---------------------------------------------------------------------------

def inject_rapid_structuring(graph, rng=None) -> dict:
    """
    Codex Seed #1 - Rapid Structuring
    scheme_type: structuring_variant
    episode_number: 0
    target_weakness: structuring detection

    Deposits made in rapid succession (< 2 hours apart) across 4 accounts,
    all just under $10k threshold. Faster timing than standard smurfing.
    """
    rng = rng or np.random.default_rng()

    individuals = [n for n, d in graph.graph.nodes(data=True) if d.get("node_type") == "individual"]
    accounts = [n for n, d in graph.graph.nodes(data=True) if d.get("node_type") == "account"]

    source = str(rng.choice(individuals))
    n_mules = 4
    mules = [str(x) for x in rng.choice([a for a in accounts if a != source], n_mules, replace=False)]
    sink = str(rng.choice([a for a in accounts if a != source and a not in mules]))

    scheme_id = f"codex_rapid_structuring_{int(rng.integers(1000, 9999))}"
    base_ts = int(rng.integers(1672531200, 1704067200))
    amounts = rng.uniform(9000.0, 9999.0, n_mules).tolist()

    edges = []
    for i, mule in enumerate(mules):
        prior = dict(graph.graph[source][mule]) if graph.graph.has_edge(source, mule) else None
        graph.graph.add_edge(source, mule,
            amount=amounts[i],
            timestamp=base_ts + i * 1800,
            transaction_type="cash_deposit",
            jurisdiction=graph.graph.nodes[source].get("jurisdiction", "US"),
            suspicion_weight=float(rng.uniform(0.85, 0.95)),
            scheme_id=scheme_id,
        )
        edges.append({"src": source, "tgt": mule, "prior": prior})

    for i, mule in enumerate(mules):
        prior = dict(graph.graph[mule][sink]) if graph.graph.has_edge(mule, sink) else None
        graph.graph.add_edge(mule, sink,
            amount=amounts[i] * float(rng.uniform(0.95, 1.0)),
            timestamp=base_ts + n_mules * 1800 + i * 3600,
            transaction_type="wire_transfer",
            jurisdiction=graph.graph.nodes[sink].get("jurisdiction", "US"),
            suspicion_weight=float(rng.uniform(0.70, 0.90)),
            scheme_id=scheme_id,
        )
        edges.append({"src": mule, "tgt": sink, "prior": prior})

    gt = {
        "source_entity": source,
        "sink_entity": sink,
        "full_path": [source] + mules + [sink],
        "scheme_type": "structuring_variant",
        "num_hops": 2,
        "intermediate_nodes": mules,
    }
    graph._injected[scheme_id] = {"edges": edges, "nodes": []}
    graph.ground_truth[scheme_id] = gt
    return gt


# ---------------------------------------------------------------------------
# Seed Scheme 2: Crypto Tumbler Chain
# ---------------------------------------------------------------------------

def inject_crypto_tumbler(graph, rng=None) -> dict:
    """
    Codex Seed #2 - Crypto Tumbler Chain
    scheme_type: crypto_variant
    episode_number: 0
    target_weakness: crypto detection

    Funds split across 3 crypto exchanges, recombined at destination.
    Fan-out/fan-in pattern makes tracing difficult.
    """
    rng = rng or np.random.default_rng()

    accounts = [n for n, d in graph.graph.nodes(data=True) if d.get("node_type") == "account"]
    cryptos = [n for n, d in graph.graph.nodes(data=True) if d.get("node_type") == "crypto_exchange"]

    source = str(rng.choice(accounts))
    n_exchanges = min(3, len(cryptos))
    exchanges = [str(x) for x in rng.choice(cryptos, n_exchanges, replace=False)]
    sink = str(rng.choice([a for a in accounts if a != source]))

    scheme_id = f"codex_crypto_tumbler_{int(rng.integers(1000, 9999))}"
    base_ts = int(rng.integers(1672531200, 1704067200))
    total_amount = float(rng.uniform(50000, 200000))
    splits = rng.dirichlet(np.ones(n_exchanges)).tolist()

    edges = []
    for i, ex in enumerate(exchanges):
        amt = total_amount * splits[i]
        prior = dict(graph.graph[source][ex]) if graph.graph.has_edge(source, ex) else None
        graph.graph.add_edge(source, ex,
            amount=amt,
            timestamp=base_ts + i * 7200,
            transaction_type="crypto_transfer",
            jurisdiction=graph.graph.nodes[ex].get("jurisdiction", "MT"),
            suspicion_weight=float(rng.uniform(0.70, 0.90)),
            scheme_id=scheme_id,
        )
        edges.append({"src": source, "tgt": ex, "prior": prior})

    for i, ex in enumerate(exchanges):
        amt = total_amount * splits[i] * float(rng.uniform(0.92, 0.99))
        prior = dict(graph.graph[ex][sink]) if graph.graph.has_edge(ex, sink) else None
        graph.graph.add_edge(ex, sink,
            amount=amt,
            timestamp=base_ts + (n_exchanges + i) * 7200,
            transaction_type="crypto_transfer",
            jurisdiction=graph.graph.nodes[sink].get("jurisdiction", "US"),
            suspicion_weight=float(rng.uniform(0.65, 0.85)),
            scheme_id=scheme_id,
        )
        edges.append({"src": ex, "tgt": sink, "prior": prior})

    gt = {
        "source_entity": source,
        "sink_entity": sink,
        "full_path": [source] + exchanges + [sink],
        "scheme_type": "crypto_variant",
        "num_hops": 2,
        "intermediate_nodes": exchanges,
    }
    graph._injected[scheme_id] = {"edges": edges, "nodes": []}
    graph.ground_truth[scheme_id] = gt
    return gt


# ---------------------------------------------------------------------------
# Seed Scheme 3: Trade Invoice Carousel
# ---------------------------------------------------------------------------

def inject_trade_carousel(graph, rng=None) -> dict:
    """
    Codex Seed #3 - Trade Invoice Carousel
    scheme_type: trade_variant
    episode_number: 0
    target_weakness: trade-based detection

    Circular trade invoicing through 3 shell companies.
    Each invoice is over/under-priced to move value.
    """
    rng = rng or np.random.default_rng()

    shells = [n for n, d in graph.graph.nodes(data=True) if d.get("node_type") == "shell_company"]
    accounts = [n for n, d in graph.graph.nodes(data=True) if d.get("node_type") == "account"]

    source = str(rng.choice(shells))
    n_mid = 3
    intermediaries = [str(x) for x in rng.choice([s for s in shells if s != source], n_mid, replace=False)]
    sink = str(rng.choice(accounts))

    scheme_id = f"codex_trade_carousel_{int(rng.integers(1000, 9999))}"
    base_ts = int(rng.integers(1672531200, 1704067200))
    base_amount = float(rng.uniform(200000, 2000000))

    path = [source] + intermediaries + [sink]
    edges = []
    for i in range(len(path) - 1):
        amt = base_amount * float(rng.uniform(0.70, 1.30))
        prior = dict(graph.graph[path[i]][path[i+1]]) if graph.graph.has_edge(path[i], path[i+1]) else None
        graph.graph.add_edge(path[i], path[i+1],
            amount=amt,
            timestamp=base_ts + i * 86400,
            transaction_type="trade_finance",
            jurisdiction=graph.graph.nodes[path[i+1]].get("jurisdiction", "HK"),
            suspicion_weight=float(rng.uniform(0.65, 0.90)),
            scheme_id=scheme_id,
        )
        edges.append({"src": path[i], "tgt": path[i+1], "prior": prior})

    gt = {
        "source_entity": source,
        "sink_entity": sink,
        "full_path": path,
        "scheme_type": "trade_variant",
        "num_hops": n_mid + 1,
        "intermediate_nodes": intermediaries,
    }
    graph._injected[scheme_id] = {"edges": edges, "nodes": []}
    graph.ground_truth[scheme_id] = gt
    return gt


# ---------------------------------------------------------------------------
# Seed Scheme 4: Offshore Layering Cascade
# ---------------------------------------------------------------------------

def inject_offshore_cascade(graph, rng=None) -> dict:
    """
    Codex Seed #4 - Offshore Layering Cascade
    scheme_type: layering_variant
    episode_number: 0
    target_weakness: layering detection

    5-hop cascade through offshore shell companies with decreasing amounts
    at each hop (mimicking fee extraction). All offshore jurisdictions.
    """
    rng = rng or np.random.default_rng()

    accounts = [n for n, d in graph.graph.nodes(data=True) if d.get("node_type") == "account"]
    shells = [n for n, d in graph.graph.nodes(data=True) if d.get("node_type") == "shell_company"]

    source = str(rng.choice(accounts))
    n_hops = 5
    intermediaries = [str(x) for x in rng.choice(shells, n_hops, replace=False)]
    sink = str(rng.choice([a for a in accounts if a != source]))

    scheme_id = f"codex_offshore_cascade_{int(rng.integers(1000, 9999))}"
    base_ts = int(rng.integers(1672531200, 1704067200))
    start_amount = float(rng.uniform(100000, 500000))
    decay = float(rng.uniform(0.93, 0.97))

    path = [source] + intermediaries + [sink]
    edges = []
    for i in range(len(path) - 1):
        amt = start_amount * (decay ** i)
        prior = dict(graph.graph[path[i]][path[i+1]]) if graph.graph.has_edge(path[i], path[i+1]) else None
        graph.graph.add_edge(path[i], path[i+1],
            amount=amt,
            timestamp=base_ts + i * 43200,
            transaction_type="swift" if i == len(path) - 2 else "wire_transfer",
            jurisdiction=graph.graph.nodes[path[i+1]].get("jurisdiction", "KY"),
            suspicion_weight=float(rng.uniform(0.75, 0.95)),
            scheme_id=scheme_id,
        )
        edges.append({"src": path[i], "tgt": path[i+1], "prior": prior})

    gt = {
        "source_entity": source,
        "sink_entity": sink,
        "full_path": path,
        "scheme_type": "layering_variant",
        "num_hops": n_hops + 1,
        "intermediate_nodes": intermediaries,
    }
    graph._injected[scheme_id] = {"edges": edges, "nodes": []}
    graph.ground_truth[scheme_id] = gt
    return gt


# ---------------------------------------------------------------------------
# Registry - built lazily so generated schemes below are defined first
# ---------------------------------------------------------------------------

def get_codex_registry():
    """Return registry mapping scheme names to callables (lazy to avoid NameError)."""
    registry = {
        "rapid_structuring": inject_rapid_structuring,
        "crypto_tumbler": inject_crypto_tumbler,
        "trade_carousel": inject_trade_carousel,
        "offshore_cascade": inject_offshore_cascade,
    }
    # Auto-discover generated inject_scheme_* functions
    import sys
    mod = sys.modules[__name__]
    for name in dir(mod):
        if name.startswith("inject_scheme_") and callable(getattr(mod, name)):
            key = name.replace("inject_scheme_", "")
            registry[key] = getattr(mod, name)
    return registry

# Backward-compatible alias
CODEX_REGISTRY = None  # Use get_codex_registry() instead

# ---------------------------------------------------------------------------
# GENERATED SCHEMES BELOW THIS LINE
# (Criminal Designer agent appends new scheme functions here)



# ---------------------------------------------------------------------------
# Generated Scheme: gen_1_crypto_variant (Episode 15)
# Target Weakness: crypto_variant
# ---------------------------------------------------------------------------

def inject_scheme_crypto_6890(graph, rng=None):
    """Codex Generated — Crypto variant 6890 with 3 exchanges."""
    import numpy as np
    rng = rng or np.random.default_rng()
    accounts = [n for n, d in graph.graph.nodes(data=True) if d.get("node_type") == "account"]
    cryptos = [n for n, d in graph.graph.nodes(data=True) if d.get("node_type") == "crypto_exchange"]
    source = str(rng.choice(accounts))
    exchanges = [str(x) for x in rng.choice(cryptos, min(3, len(cryptos)), replace=False)]
    sink = str(rng.choice([a for a in accounts if a != source]))
    scheme_id = f"codex_crypto_6890_{int(rng.integers(1000,9999))}"
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
        edges.append({"src": path[i], "tgt": path[i+1], "prior": prior})
    gt = {"source_entity": source, "sink_entity": sink, "full_path": path,
           "scheme_type": "crypto_variant", "num_hops": len(exchanges) + 1, "intermediate_nodes": exchanges}
    graph._injected[scheme_id] = {"edges": edges, "nodes": []}
    graph.ground_truth[scheme_id] = gt
    return gt



# ---------------------------------------------------------------------------
# Generated Scheme: gen_2_trade_variant (Episode 15)
# Target Weakness: trade_variant
# ---------------------------------------------------------------------------

def inject_scheme_trade_1773(graph, rng=None):
    """Codex Generated — Trade variant 1773 with 4 intermediaries."""
    import numpy as np
    rng = rng or np.random.default_rng()
    shells = [n for n, d in graph.graph.nodes(data=True) if d.get("node_type") == "shell_company"]
    accounts = [n for n, d in graph.graph.nodes(data=True) if d.get("node_type") == "account"]
    source = str(rng.choice(shells))
    intermediaries = [str(x) for x in rng.choice([s for s in shells if s != source], min(4, len(shells)-1), replace=False)]
    sink = str(rng.choice(accounts))
    scheme_id = f"codex_trade_1773_{int(rng.integers(1000,9999))}"
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
        edges.append({"src": path[i], "tgt": path[i+1], "prior": prior})
    gt = {"source_entity": source, "sink_entity": sink, "full_path": path,
           "scheme_type": "trade_variant", "num_hops": 4 + 1, "intermediate_nodes": intermediaries}
    graph._injected[scheme_id] = {"edges": edges, "nodes": []}
    graph.ground_truth[scheme_id] = gt
    return gt



# ---------------------------------------------------------------------------
# Generated Scheme: gen_1_trade_variant (Episode 0)
# Target Weakness: trade_variant
# ---------------------------------------------------------------------------

def inject_scheme_trade_2567(graph, rng=None):
    """Codex Generated — Trade variant 2567 with 2 intermediaries."""
    import numpy as np
    rng = rng or np.random.default_rng()
    shells = [n for n, d in graph.graph.nodes(data=True) if d.get("node_type") == "shell_company"]
    accounts = [n for n, d in graph.graph.nodes(data=True) if d.get("node_type") == "account"]
    source = str(rng.choice(shells))
    intermediaries = [str(x) for x in rng.choice([s for s in shells if s != source], min(2, len(shells)-1), replace=False)]
    sink = str(rng.choice(accounts))
    scheme_id = f"codex_trade_2567_{int(rng.integers(1000,9999))}"
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
        edges.append({"src": path[i], "tgt": path[i+1], "prior": prior})
    gt = {"source_entity": source, "sink_entity": sink, "full_path": path,
           "scheme_type": "trade_variant", "num_hops": 2 + 1, "intermediate_nodes": intermediaries}
    graph._injected[scheme_id] = {"edges": edges, "nodes": []}
    graph.ground_truth[scheme_id] = gt
    return gt



# ---------------------------------------------------------------------------
# Generated Scheme: gen_1_structuring_variant (Episode 5)
# Target Weakness: structuring_variant
# ---------------------------------------------------------------------------

def inject_scheme_struct_6890(graph, rng=None):
    """Codex Generated — Structuring variant 6890 with 5 mules."""
    import numpy as np
    rng = rng or np.random.default_rng()
    individuals = [n for n, d in graph.graph.nodes(data=True) if d.get("node_type") == "individual"]
    accounts = [n for n, d in graph.graph.nodes(data=True) if d.get("node_type") == "account"]
    source = str(rng.choice(individuals))
    mules = [str(x) for x in rng.choice([a for a in accounts if a != source], 5, replace=False)]
    sink = str(rng.choice([a for a in accounts if a != source and a not in mules]))
    scheme_id = f"codex_struct_6890_{int(rng.integers(1000,9999))}"
    base_ts = int(rng.integers(1672531200, 1704067200))
    amounts = rng.uniform(8500.0, 9999.0, 5).tolist()
    edges = []
    for i, m in enumerate(mules):
        prior = dict(graph.graph[source][m]) if graph.graph.has_edge(source, m) else None
        graph.graph.add_edge(source, m, amount=amounts[i], timestamp=base_ts + i * int(rng.integers(900, 7200)),
            transaction_type="cash_deposit", jurisdiction=graph.graph.nodes[source].get("jurisdiction", "US"),
            suspicion_weight=float(rng.uniform(0.80, 0.95)), scheme_id=scheme_id)
        edges.append({"src": source, "tgt": m, "prior": prior})
    for i, m in enumerate(mules):
        prior = dict(graph.graph[m][sink]) if graph.graph.has_edge(m, sink) else None
        graph.graph.add_edge(m, sink, amount=amounts[i] * float(rng.uniform(0.93, 1.0)),
            timestamp=base_ts + 5 * 7200 + i * 3600, transaction_type="wire_transfer",
            jurisdiction=graph.graph.nodes[sink].get("jurisdiction", "US"),
            suspicion_weight=float(rng.uniform(0.65, 0.90)), scheme_id=scheme_id)
        edges.append({"src": m, "tgt": sink, "prior": prior})
    gt = {"source_entity": source, "sink_entity": sink, "full_path": [source] + mules + [sink],
           "scheme_type": "structuring_variant", "num_hops": 2, "intermediate_nodes": mules}
    graph._injected[scheme_id] = {"edges": edges, "nodes": []}
    graph.ground_truth[scheme_id] = gt
    return gt



# ---------------------------------------------------------------------------
# Generated Scheme: gen_2_crypto_variant (Episode 10)
# Target Weakness: crypto_variant
# ---------------------------------------------------------------------------

def inject_scheme_crypto_1773(graph, rng=None):
    """Codex Generated — Crypto variant 1773 with 4 exchanges."""
    import numpy as np
    rng = rng or np.random.default_rng()
    accounts = [n for n, d in graph.graph.nodes(data=True) if d.get("node_type") == "account"]
    cryptos = [n for n, d in graph.graph.nodes(data=True) if d.get("node_type") == "crypto_exchange"]
    source = str(rng.choice(accounts))
    exchanges = [str(x) for x in rng.choice(cryptos, min(4, len(cryptos)), replace=False)]
    sink = str(rng.choice([a for a in accounts if a != source]))
    scheme_id = f"codex_crypto_1773_{int(rng.integers(1000,9999))}"
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
        edges.append({"src": path[i], "tgt": path[i+1], "prior": prior})
    gt = {"source_entity": source, "sink_entity": sink, "full_path": path,
           "scheme_type": "crypto_variant", "num_hops": len(exchanges) + 1, "intermediate_nodes": exchanges}
    graph._injected[scheme_id] = {"edges": edges, "nodes": []}
    graph.ground_truth[scheme_id] = gt
    return gt



# ---------------------------------------------------------------------------
# Generated Scheme: gen_1_trade_variant (Episode 0)
# Target Weakness: trade_variant
# ---------------------------------------------------------------------------

def inject_scheme_trade_2567(graph, rng=None):
    """Codex Generated — Trade variant 2567 with 2 intermediaries."""
    import numpy as np
    rng = rng or np.random.default_rng()
    shells = [n for n, d in graph.graph.nodes(data=True) if d.get("node_type") == "shell_company"]
    accounts = [n for n, d in graph.graph.nodes(data=True) if d.get("node_type") == "account"]
    source = str(rng.choice(shells))
    intermediaries = [str(x) for x in rng.choice([s for s in shells if s != source], min(2, len(shells)-1), replace=False)]
    sink = str(rng.choice(accounts))
    scheme_id = f"codex_trade_2567_{int(rng.integers(1000,9999))}"
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
        edges.append({"src": path[i], "tgt": path[i+1], "prior": prior})
    gt = {"source_entity": source, "sink_entity": sink, "full_path": path,
           "scheme_type": "trade_variant", "num_hops": 2 + 1, "intermediate_nodes": intermediaries}
    graph._injected[scheme_id] = {"edges": edges, "nodes": []}
    graph.ground_truth[scheme_id] = gt
    return gt



# ---------------------------------------------------------------------------
# Generated Scheme: gen_1_structuring_variant (Episode 5)
# Target Weakness: structuring_variant
# ---------------------------------------------------------------------------

def inject_scheme_struct_6890(graph, rng=None):
    """Codex Generated — Structuring variant 6890 with 5 mules."""
    import numpy as np
    rng = rng or np.random.default_rng()
    individuals = [n for n, d in graph.graph.nodes(data=True) if d.get("node_type") == "individual"]
    accounts = [n for n, d in graph.graph.nodes(data=True) if d.get("node_type") == "account"]
    source = str(rng.choice(individuals))
    mules = [str(x) for x in rng.choice([a for a in accounts if a != source], 5, replace=False)]
    sink = str(rng.choice([a for a in accounts if a != source and a not in mules]))
    scheme_id = f"codex_struct_6890_{int(rng.integers(1000,9999))}"
    base_ts = int(rng.integers(1672531200, 1704067200))
    amounts = rng.uniform(8500.0, 9999.0, 5).tolist()
    edges = []
    for i, m in enumerate(mules):
        prior = dict(graph.graph[source][m]) if graph.graph.has_edge(source, m) else None
        graph.graph.add_edge(source, m, amount=amounts[i], timestamp=base_ts + i * int(rng.integers(900, 7200)),
            transaction_type="cash_deposit", jurisdiction=graph.graph.nodes[source].get("jurisdiction", "US"),
            suspicion_weight=float(rng.uniform(0.80, 0.95)), scheme_id=scheme_id)
        edges.append({"src": source, "tgt": m, "prior": prior})
    for i, m in enumerate(mules):
        prior = dict(graph.graph[m][sink]) if graph.graph.has_edge(m, sink) else None
        graph.graph.add_edge(m, sink, amount=amounts[i] * float(rng.uniform(0.93, 1.0)),
            timestamp=base_ts + 5 * 7200 + i * 3600, transaction_type="wire_transfer",
            jurisdiction=graph.graph.nodes[sink].get("jurisdiction", "US"),
            suspicion_weight=float(rng.uniform(0.65, 0.90)), scheme_id=scheme_id)
        edges.append({"src": m, "tgt": sink, "prior": prior})
    gt = {"source_entity": source, "sink_entity": sink, "full_path": [source] + mules + [sink],
           "scheme_type": "structuring_variant", "num_hops": 2, "intermediate_nodes": mules}
    graph._injected[scheme_id] = {"edges": edges, "nodes": []}
    graph.ground_truth[scheme_id] = gt
    return gt



# ---------------------------------------------------------------------------
# Generated Scheme: gen_2_crypto_variant (Episode 10)
# Target Weakness: crypto_variant
# ---------------------------------------------------------------------------

def inject_scheme_crypto_1773(graph, rng=None):
    """Codex Generated — Crypto variant 1773 with 4 exchanges."""
    import numpy as np
    rng = rng or np.random.default_rng()
    accounts = [n for n, d in graph.graph.nodes(data=True) if d.get("node_type") == "account"]
    cryptos = [n for n, d in graph.graph.nodes(data=True) if d.get("node_type") == "crypto_exchange"]
    source = str(rng.choice(accounts))
    exchanges = [str(x) for x in rng.choice(cryptos, min(4, len(cryptos)), replace=False)]
    sink = str(rng.choice([a for a in accounts if a != source]))
    scheme_id = f"codex_crypto_1773_{int(rng.integers(1000,9999))}"
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
        edges.append({"src": path[i], "tgt": path[i+1], "prior": prior})
    gt = {"source_entity": source, "sink_entity": sink, "full_path": path,
           "scheme_type": "crypto_variant", "num_hops": len(exchanges) + 1, "intermediate_nodes": exchanges}
    graph._injected[scheme_id] = {"edges": edges, "nodes": []}
    graph.ground_truth[scheme_id] = gt
    return gt

