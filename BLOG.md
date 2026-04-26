# HEIST: I Built an AI Detective That Fights an AI Criminal — And the Criminal Fights Back

*A co-evolving adversarial RL environment for financial crime investigation*

---

## The Problem

Money laundering costs the global economy **$2 trillion a year**. Every existing detection system runs on static rules. The moment a criminal figures out the rules, they route around them. It's a cat-and-mouse game — and right now, the mouse is winning.

What if the cat could learn? What if the criminal could evolve?

That's HEIST.

---

## What I Built

HEIST is a **multi-agent adversarial reinforcement learning environment** where two AIs go head to head:

- 🕵️ **The Investigator** — a Qwen2.5-7B model trained with GRPO. It gets 6 investigation tools (query entity, trace transactions, cross-reference, flag suspicious, request records, file SAR) and up to 50 steps to find the criminal in a 100,000-node financial transaction graph.

- 💀 **The Criminal Designer** — an LLM that *writes Python code* to generate novel money laundering schemes. It doesn't pick from a list. It generates new attack patterns every K episodes, targeting whatever the investigator is worst at.

Neither agent has hardcoded knowledge. Both learn from each other's failures.

---

## The Key Idea: AdversarialCodex

The most reusable piece of HEIST is something I'm calling **AdversarialCodex** — a domain-agnostic adversarial curriculum generator that any OpenEnv environment can plug in.

```python
codex = AdversarialCodex(
    env_description="financial crime investigation",
    scheme_types=["smurfing", "layering", "crypto_mixing"],
    llm_call_fn=my_llm,
    novelty_threshold=0.05,
)

# After each episode — track what the agent struggles with
codex.update(scheme_type="crypto_mixing", investigator_f1=0.25)

# Every K episodes — generate a harder variant of the worst scheme
new_attack = codex.generate(target_weakness="crypto_mixing")

# At the end — synthesize a Zero-Day: a composite of the 3 hardest attacks
zero_day = codex.synthesize_zero_day(top_n=3)
```

Three properties make it powerful:

1. **ELO-tracked difficulty** — every scheme type has an ELO rating. When the investigator fails, the scheme's ELO rises. The curriculum samples hardest schemes most often.
2. **Novelty-gated generation** — new schemes are only accepted if they're structurally different from existing ones (cosine distance check). No duplicate attacks.
3. **Zero-Day synthesis** — at the end of training, the top-3 hardest schemes are combined into a single composite attack the investigator has never seen.

---

## What Else Makes This Different

**Hindsight Experience Replay for LLM agents** — the first application of HER to tool-use LLM agents. GRPO suffers badly from sparse rewards: the investigator only gets a signal when it files a SAR at step 30-50. HER retroactively relabels failed episodes: *"You didn't catch the real criminal — but if the criminal had been at entity X, your investigation was correct."* Every failure becomes a learning signal.

**Non-stationary reward (Preference Drift)** — the Compliance Expert evaluating SARs changes what it values across three phases:
- Episodes 0–20: thoroughness (verbose evidence chains)
- Episodes 20–50: precision (only relevant entities)
- Episodes 50+: speed + accuracy (minimal queries, tight chain)

The investigator must adapt to an evaluator whose preferences shift over time.

**Bayesian belief updating** — the investigator maintains a real-time probability distribution P(criminal | evidence) over all 100k entities. Every query updates beliefs via likelihood ratios (3.5× for suspicious signals, 0.75× for clean). It always knows its current best suspect.

**Mid-episode criminal morphing** — when P(criminal) > 0.7 for any entity, the criminal reroutes funds through new shell companies mid-episode, invalidating the investigator's evidence chain. This is a real Stackelberg game.

**Shapley attribution** — across the 3-agent coalition (Investigator, Compliance Expert, Oversight), exact Shapley values tell each agent its fair share of the final reward.

---

## Training Results

I trained Qwen2.5-7B-Instruct (4-bit QLoRA) on an A100 for 44 episodes using GRPO with G=8 rollouts.

| Metric | Value |
|--------|-------|
| Best F1 | **0.571** |
| Mean F1 | 0.40 |
| R_inv range | 0.097 – 0.249 |
| crypto_mixing ELO | **1292** (started at 1200) |

The Red Queen curriculum is working. `crypto_mixing` climbed to ELO 1292 because it consistently beats the investigator. The curriculum now samples it 3× more than other schemes, forcing the agent to get better at exactly its hardest challenge.

The Zero-Day synthesized at the end combines `crypto_mixing + trade_based + smurfing` — a composite attack no human designed, engineered purely from the investigator's own failure modes.

---

## Try It

**Live demo**: https://huggingface.co/spaces/Rohan333555/heist-demo

Click **"⚡ Simulate 10 Steps"** in the sidebar to run a live investigation episode. Watch the Bayesian belief distribution update in real time as the investigator queries entities, traces transactions, and builds its evidence chain.

**Code**: https://github.com/RohanKar3535/heist

```bash
pip install uv
cd env && uv sync
python inference.py --direct --seed 42
```

---

## Why This Matters Beyond HEIST

AdversarialCodex is infrastructure, not just a project. Any OpenEnv environment dealing with an adversarial or evolving task can drop it in — fraud detection, cybersecurity, content moderation, game AI. The ELO curriculum, novelty gating, and Zero-Day synthesis are domain-agnostic.

HEIST is one environment. AdversarialCodex is a primitive for the whole ecosystem.

---

*Built for the Meta × HuggingFace × PyTorch OpenEnv Hackathon 2026*  
*GitHub: [RohanKar3535/heist](https://github.com/RohanKar3535/heist)*  
*Demo: [Rohan333555/heist-demo](https://huggingface.co/spaces/Rohan333555/heist-demo)*
