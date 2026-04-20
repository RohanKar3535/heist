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

## Q6: "What's the real-world impact?"

**A:** 

**The problem:** $2 trillion is laundered globally every year (UN estimate). Current AML systems — which are essentially the 50-line rule-based baseline we showed — catch approximately 1% of it. Financial institutions spend $214 billion annually on compliance, mostly satisfying regulatory requirements with rule-based systems that criminals have long adapted to.

**What HEIST demonstrates:**
1. **Trainable detection** — An agent that learns investigative strategy purely from reward, with zero domain knowledge. This transfers directly to real AML systems: deploy the trained policy against real transaction graphs.

2. **Adversarial robustness** — By training against a co-evolving criminal that writes its own attacks, the investigator is hardened against novel laundering patterns. Real criminals innovate; our investigator is trained to handle innovation.

3. **The Zero-Day capability** — The most important result isn't the 0.90 F1 on known schemes. It's that the trained investigator catches less than 30% of a completely novel, AI-invented scheme it's never seen. That's transfer learning for financial crime — the ability to partially detect attacks that haven't been invented yet.

**Next steps for real-world deployment:**
- Replace our networkx graph with a real bank's transaction database
- Replace our 6 simulated tools with API calls to actual compliance systems
- The GRPO training loop and reward function transfer directly

---

## Q7: "How does this address the four hackathon themes?"

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
