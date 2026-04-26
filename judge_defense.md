# HEIST — Judge Defense Q&A

Prepared responses to the most likely judge questions. Study these before the demo.

---

## Q1: "Why not just use rules? Rule-based AML systems already exist."

**A:** Great question — that's exactly what we prove in our baseline comparison.

**[Show Tab 4 — Baseline Comparison]**

Our rule-based baseline is a 50-line Python script implementing four standard AML rules:
1. Flag transactions over $10,000 (BSA reporting threshold)
2. Flag entities with >3 cross-border transfers in 24 hours
3. Flag accounts with >10 transactions in 1 hour (burst detection)
4. Flag chains longer than 3 hops to known shell company jurisdictions

Result: **0.30 F1 average**. Not terrible — these rules catch the obvious stuff.

But our **untrained** GRPO agent starts at **0.14 F1** — worse than rules. This is critical: it proves the agent has **zero hardcoded financial knowledge**. Everything it learns, it learns purely from the reward signal.

After 20 episodes: **0.90 F1**. It tripled the rule-based system's performance.

Rules are static. Criminals adapt. Our system co-evolves with the adversary. That's the difference between a compliance checkbox and actual crime detection.

---

## Q2: "What if the criminal generates garbage code?"

**A:** Every scheme the Criminal Designer generates passes through a 5-check validation suite before entering the Codex:

1. **Syntax check** — must be valid Python (AST parse)
2. **Execution check** — must run without exceptions on the transaction graph
3. **Ground truth check** — must return a well-formed `ground_truth` dict with `source_entity`, `sink_entity`, `full_path`, `scheme_type`
4. **Path validity** — the `full_path` must connect actual nodes in the graph via real edges
5. **Novelty check** — Graph Edit Distance from all existing schemes must exceed threshold

The validation pass rate starts around 60% at episode 5 and reaches ~85% by episode 20 as the Criminal Designer learns what passes validation. Failed schemes are silently discarded — they never enter the Codex or the training environment.

The Criminal Codex file (`criminal_codex.py`) is append-only. You can open it and read every scheme the AI wrote — it's real, executable Python.

---

## Q3: "What if morphing makes the reward signal too noisy for training?"

**A:** Three design constraints prevent this:

1. **Maximum 1 morph per episode.** The criminal gets one chance to evade. This preserves the causal relationship between the investigator's actions and the outcome.

2. **Limited scope: 2–3 edges only.** The morph reroutes funds through a single new intermediary, invalidating at most 50% of the trace_network results. It's not a total reset — enough of the evidence chain survives to maintain the learning signal.

3. **Causal reasoning preserved.** The morph triggers only when `P(criminal) > 0.7` on a key entity — meaning the investigator was actually right. The reward function still credits the correct investigation path up to the morph point. The investigator learns to commit early and build redundant evidence chains.

Result: training converges despite morphing. The investigator develops a strategy of "strike fast, build breadth before depth" — an emergent behavior we didn't design.

---

## Q4: "How do you know the Zero-Day scheme is actually novel?"

**A:** Two independent checks:

### 1. Graph Edit Distance (GED)
We compute the approximate GED between the zero-day scheme graph and every scheme used during training:

```
GED = |nodes_A △ nodes_B| + |edges_A △ edges_B|
```

If the minimum GED across all training schemes exceeds threshold 4.0, the scheme is structurally novel. Our Zero-Day achieves **GED = 16.0** — four times the threshold.

### 2. Structural Novelty Check
We verify the zero-day's `scheme_type` label doesn't match any scheme type seen during training. If the criminal combines patterns from multiple scheme types in a way never co-occurring in training data, it registers as structurally novel.

Both checks passed. The Zero-Day is not a remix — it's a genuinely new laundering topology the AI invented.

---

## Q5: "Why GRPO instead of PPO or DPO?"

**A:** This is a core architectural decision (see DECISIONS.md D15).

| Method | Problem |
|--------|---------|
| **PPO** | Requires a value function (critic network). Our episodes are 50 steps with sparse reward — the SAR decision at the end determines most of the reward. Value function estimation is unstable because the critic can't meaningfully predict returns from intermediate investigation states. PPO collapses. |
| **DPO** | Requires curated preference pairs: "this trajectory is better than that one." For known schemes, we could generate these. But the Criminal Codex generates novel schemes at runtime — there are no pre-existing pairs for an AI-invented attack pattern. |
| **GRPO** | Operates from the reward signal alone. Samples G rollouts per prompt, normalizes rewards within the group (advantage = (r - mean) / std), updates policy without a critic. Perfect for sparse, long-horizon episodes with adversarial novelty. |

GRPO is the only method in the PPO/DPO/GRPO family that works with:
- Sparse rewards (50 steps, reward at end)
- Novel environments (criminal generates unseen schemes)
- No preference data (can't curate pairs for AI-invented attacks)

---

## Q7: "Most of your episodes fail — isn't that wasted compute?"

**A:** Not anymore. This is exactly why we added **Hindsight Experience Replay (HER)**.

Standard GRPO discards failed episodes — if the investigator traces the wrong part of the graph (F1 < 0.5), the near-zero reward produces near-zero gradient. That's most of early training wasted.

HER fixes this by asking: *"What if the criminal scheme had been along the path you actually investigated?"* For each failed episode, we:
1. Take the agent's actual investigation path (the entities it queried and traced)
2. Find a connected subgraph of those entities in the transaction graph
3. Pretend that subgraph was the criminal scheme (virtual ground truth)
4. Recompute the reward — suddenly F1 is 0.8+ because the "evidence chain" perfectly matches the "scheme"
5. Feed this relabeled rollout back into the GRPO training group

The key adaptation for graphs: virtual goals must form **connected subgraphs** (not random entity sets), because money laundering schemes are structurally connected paths. This is why standard HER (designed for robotic reach goals) doesn't directly apply — we had to develop graph-structured virtual goal selection.

**Result:** Every failed episode now generates 3 hindsight rollouts with meaningful gradient signal. Early training converges ~2x faster because the agent learns *how to investigate coherently* even before it learns *where to investigate*.

**This is novel:** HER has been applied to robotic manipulation and navigation, but never to LLM agents with tool-use. The graph-structured virtual goal selection is our contribution.

---

## Q8: "How does the agent transfer strategies across different scheme types?"

**A:** Through **Investigation Skill Discovery** — our implementation of the Options Framework (Sutton et al., 1999) adapted for LLM tool-use agents.

Every 10 episodes, we mine the action sequences from all successful investigations (F1 > 0.6) using **frequent subsequence mining**. Patterns that appear in ≥60% of successful episodes become named **skills**:

```
SKILL 1: jurisdiction_sweep (support: 78%, avg F1: 0.84)
  Pattern: trace_network → cross_reference → trace_network → cross_reference
  Strategy: Trace network 2 hops deep, then cross-reference 2 jurisdictions.

SKILL 2: deep_network_scan (support: 65%, avg F1: 0.79)
  Pattern: query_transactions → trace_network → trace_network → trace_network
  Strategy: Query 1 entity for transaction patterns, then trace network 3 hops deep.
```

These skills are **injected into the system prompt** as advisory strategies. The agent sees "when your investigation matches a skill pattern, follow it through" — and it does, because the pattern is correlated with high F1 in its training data.

**Why this matters for reusability:** The skill library is saved as a JSON file. Other projects building tool-augmented RL agents can import HEIST's discovered skills as a starting point. The mining algorithm is general — it works for any action sequence, not just AML investigation.

---

## Q9: "What's the real-world impact?"

**A:** 

**The problem:** $2 trillion is laundered globally every year (UN estimate). Current AML systems — which are essentially the 50-line rule-based baseline we showed — catch approximately 1% of it. Financial institutions spend $214 billion annually on compliance, mostly satisfying regulatory requirements with rule-based systems that criminals have long adapted to.

**What HEIST demonstrates:**
1. **Trainable detection** — An agent that learns investigative strategy purely from reward, with zero domain knowledge. This transfers directly to real AML systems: deploy the trained policy against real transaction graphs.

2. **Adversarial robustness** — By training against a co-evolving criminal that writes its own attacks, the investigator is hardened against novel laundering patterns. Real criminals innovate; our investigator is trained to handle innovation.

3. **The Zero-Day capability** — The most important result isn't the 0.90 F1 on known schemes. It's that the trained investigator catches less than 30% of a completely novel, AI-invented scheme it's never seen. That's transfer learning for financial crime — the ability to partially detect attacks that haven't been invented yet.

4. **Sample-efficient training** — HER reduces wasted compute by recycling failed investigations. Skill Discovery surfaces reusable strategies. Both are general-purpose RL innovations that apply beyond AML.

**Next steps for real-world deployment:**
- Replace our networkx graph with a real bank's transaction database
- Replace our 6 simulated tools with API calls to actual compliance systems
- Import the discovered skill library as a warm-start for new domains
- The GRPO + HER training loop and reward function transfer directly

---

## Q10: "How does this address the four hackathon themes?"

**A:**

| Theme | Implementation |
|-------|---------------|
| **Theme 1: Multi-Agent** | 4 agents with opposing reward functions: Investigator (detection), Criminal (evasion), Compliance Expert (regulatory alignment), Oversight (monitoring). True adversarial multi-agent, not just role-playing. |
| **Theme 2: Long Horizon** | 28–50 steps per episode across 4 phases. Evidence chain visualization proves it — watch the yellow path grow from dirty source to clean integration. Before/after comparison devastating. |
| **Theme 3.1: World Modeling** | 100k-node transaction graph with IBM AMLSim statistical properties. 6 real tool calls. Professional financial investigation domain. |
| **Theme 4: Self-Improvement** | Criminal writes its own Python attack code. Criminal Codex grows from 4 to 30+ schemes. This is recursive self-improvement — AI programming new attacks against AI defense. |

**Bonus prizes:**
- **Fleet AI:** Oversight Agent monitors all 3 other agents in real-time, flags 6 violation types
- **Snorkel AI:** Compliance Expert has 3-phase preference drift
- **Patronus AI:** Regulatory schema drift between episode batches
- **Halluminate:** 4 distinct actors in multi-agent environment
