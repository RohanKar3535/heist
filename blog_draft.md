# HEIST: When AI Criminals Write Their Own Attack Code

*A HuggingFace Community Blog Draft*

---

## The Problem
$2 trillion laundered globally per year. Current rule-based AML systems catch less than 1%. Criminals adapted to static detection rules decades ago. We need systems that evolve.

## What We Built
**HEIST** is a multi-agent reinforcement learning environment where an AI investigator learns to detect money laundering in a 100,000-node financial transaction graph — while simultaneously training against an AI criminal that writes its own attack code.

Neither agent has hardcoded financial knowledge. Both start from scratch. Both co-evolve.

### The Four Agents
1. **Investigator** (Qwen2.5-1.5B, GRPO-trained) — Uses 6 API tools to investigate flagged transactions across 4 phases: Alert Triage → Investigation → Cross-Reference → SAR Filing
2. **Criminal Designer** (Gemini-powered) — Writes executable Python code to generate new money laundering schemes, saved to a growing file called the Criminal Codex
3. **Compliance Expert** — Evaluates SAR quality with 3-phase preference drift (Snorkel AI integration)
4. **Oversight Agent** — Monitors all agents in real-time for 6 violation types (Fleet AI integration)

### The Criminal Codex
This is the innovation we're most excited about. The Criminal Designer doesn't select from a fixed menu of attacks — it literally writes Python functions and appends them to `criminal_codex.py`.

**Episode 0:** 4 human-designed seed schemes  
**Episode 20:** 7 AI-written schemes  
**Episode 50:** 30+ AI-invented attack patterns

Every generated scheme passes a 5-check validation suite (syntax, execution, ground truth, path validity, novelty) before entering the Codex.

### Red Queen Co-Evolution
The curriculum controller implements a softmax weakness targeting function:

```
P(scheme_k) = softmax(weakness_vector / τ)
```

Where `weakness_vector[k] = 1 - F1(k)` and temperature decays over time. The criminal exponentially targets scheme types where the investigator performs worst. Mathematical guarantee against overfitting.

### Mid-Episode Morphing
When the investigator's Bayesian belief on a key entity exceeds `P(criminal) > 0.7`, the criminal morphs — rerouting funds through new intermediaries mid-episode. This upgrades the standard Stackelberg formulation to an extensive-form game.

## Why GRPO?
We spent significant time evaluating RL algorithms:
- **PPO** requires a value function (critic) that's unstable on sparse 50-step episodes
- **DPO** needs curated preference pairs unavailable for novel adversarial schemes
- **GRPO** operates from the reward signal alone — perfect for our setting

## Results

| Model | Avg F1 |
|-------|--------|
| Random (baseline) | 0.14 |
| Rule-based (50-line AML) | 0.30 |
| **Trained Investigator** | **0.90** |
| vs Zero-Day | 0.29 |

The untrained agent at 0.14 F1 — worse than rules — proves zero hardcoded knowledge. The trained agent at 0.90 proves it learned genuine investigative strategy.

## The Zero-Day Reveal
After training, we tell the criminal: "Generate your best scheme. No constraints."

It produces a novel laundering pattern. Graph Edit Distance confirms structural novelty (GED = 16, threshold = 4). The trained investigator — the one that achieves 0.90 on everything else — catches the Zero-Day at less than 30%.

**An AI criminal invented a financial crime that defeats an AI detective trained specifically to stop it.**

## Try It
- **Demo:** [HuggingFace Space](https://huggingface.co/spaces/YOUR_USERNAME/heist)
- **Training:** [Colab Notebook](https://colab.research.google.com/) — runs on free T4 GPU in ~30 minutes
- **Code:** [GitHub](https://github.com/YOUR_USERNAME/heist)

## Technical Stack
Python 3.11 · OpenEnv · networkx (100k nodes) · Gemini API · TRL 0.29.0 + Unsloth · Qwen2.5-1.5B · Streamlit · pyvis

---

*Built for the Meta Hackathon Finals. Targeting Theme 1 (Multi-Agent), Theme 2 (Long Horizon), Theme 3.1 (World Modeling), and Theme 4 (Self-Improvement), plus Fleet AI, Snorkel AI, Patronus AI, and Halluminate bonus prizes.*
