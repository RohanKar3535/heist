# HEIST — 3-Minute Demo Script

> Exact pitch for hackathon judges. Deliver word-for-word.

---

## [0:00–0:30] HOOK

*"Two trillion dollars. That's how much money is laundered globally every year. And we catch less than one percent of it."*

*"Current anti-money-laundering systems are rule-based — flag transactions over ten thousand dollars, check a sanctions list, file a report. Criminals adapted to these rules decades ago."*

*"We built HEIST: an AI investigator that learns to detect money laundering in a hundred-thousand-node financial network. It has zero hardcoded financial knowledge. And it's trained against an AI criminal that writes its own attack code."*

---

## [0:30–1:15] THE ENVIRONMENT

*"Four agents. One graph. A hundred thousand nodes."*

**[Show Transaction Network on Tab 1]**

*"The investigator starts here — a flagged transaction. It has six tools: query transactions, trace the network, cross-reference jurisdictions, request subpoenas, and ultimately file a Suspicious Activity Report. Fifty steps. Four phases. No hardcoded strategy."*

*"On the other side: the Criminal Designer. Powered by Gemini, it doesn't just select attacks from a menu — it writes actual executable Python code and saves it to a file called the Criminal Codex."*

**[Show Criminal Codex on Tab 2]**

*"It starts with four human-designed seed schemes. By episode twenty, it contains seven AI-written schemes. By episode fifty, over thirty. The criminal targets the investigator's weakest detection — if you're bad at catching shell chains, that's exactly what it generates next."*

*"And if the investigator gets too close — Bayesian belief exceeds 0.7 on a key entity — the criminal morphs. It reroutes funds mid-episode, invalidates the evidence chain. This is an extensive-form Stackelberg game."*

---

## [1:15–2:00] RED QUEEN CO-EVOLUTION

**[Switch to Tab 2 — Red Queen Battle]**

*"This is the Red Queen effect. Both agents co-evolve. The criminal's ELO rises when it evades detection. The investigator's ELO rises when it catches novel schemes."*

**[Point to weakness heatmap]**

*"The curriculum controller tracks the investigator's F1 per scheme type. Where it's weakest, the criminal strikes hardest. Softmax probability with decaying temperature — mathematical guarantee the investigator can't overfit to any single scheme."*

**[Switch to Tab 3 — Training Curves]**

*"Here's the training curve. We use GRPO — not PPO, not DPO. PPO requires a value function that collapses on sparse fifty-step episodes where reward only arrives at SAR filing. DPO needs curated preference pairs that don't exist for novel adversarial schemes. GRPO operates from the reward signal alone."*

*"Detection F1: starts at 0.14 — worse than the fifty-line rule baseline at 0.30. That proves zero hardcoded knowledge. By episode twenty, it reaches 0.90. It learned genuine investigative strategy."*

---

## [2:00–2:45] THE MATH

**[Switch to Tab 4 — Baseline Comparison]**

*"Four models compared. Random baseline: 0.14 F1 — that's our starting point, proving the agent has zero prior knowledge. Rule-based: 0.30 — flag amounts over ten thousand, check for burst transactions. Our trained investigator: 0.90. And against the Zero-Day..."*

*"The advanced mathematics driving this: Bayesian belief updating on entity criminality — each query updates posterior probability across the network. Information-theoretic query selection maximizes mutual information per tool call. Red Queen curriculum uses softmax weakness targeting with temperature decay τ equals τ-zero times 0.95 to the power of episode over K. Graph edit distance measures structural novelty."*

---

## [2:45–3:00] ZERO-DAY REVEAL

**[Switch to Tab 5 — Zero-Day Reveal]**

*"After training, we tell the criminal: generate your best scheme. No constraints. No rules."*

*"It produces a novel laundering pattern that no human designed. Graph edit distance confirms structural novelty — GED of sixteen, well above the threshold of four."*

*"Our trained investigator — the one that crushes everything at 0.90 F1 — catches this Zero-Day at less than thirty percent."*

**[Point to closing pitch box]**

*"This is not a demo of a chatbot. This is a co-evolving adversarial system where an AI criminal invented a new financial crime pattern that defeats an AI detective trained specifically to stop it."*

**[Pause. Look at judges.]**

*"Two trillion dollars. We need systems that can find what rules cannot. HEIST proves that's possible."*

---

## Timing Notes
- Practice the hook until it's under 30 seconds
- Tab switches should be pre-loaded — click tab before starting the next section
- The Zero-Day pause is critical — let it land
- If judges interrupt with questions, redirect: "Great question — I have a defense document for that, happy to go deep after the pitch"
