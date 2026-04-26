# HEIST — Master Plan

## One-Line Pitch
"An AI detective and an AI criminal. Neither has hardcoded knowledge. The criminal writes its own attack code. Both co-evolve for 50 episodes. At the end the criminal generates a novel money laundering pattern that no human designed — and our trained investigator still can't fully catch it."

## What This Is
OpenEnv RL environment for Meta hackathon finals. Financial crime investigation domain. Agent learns to detect money laundering in a dynamic transaction network. Adversarial criminal co-evolves simultaneously and writes its own attack code. 4 agents total. 4 themes. 4 bonus prizes.

## Stack
- Python 3.11
- OpenEnv (latest release) — pyproject.toml + uv
- FastAPI (OpenEnv server)
- networkx (transaction graph)
- Gemini API (Criminal + Expert + Oversight LLM calls)
- TRL 0.29.0 + Unsloth (GRPO training)
- Qwen2.5-1.5B (base model for training)
- Streamlit (Investigation War Room UI)
- pyvis (transaction network visualization)
- HuggingFace Spaces (deployment)

## The 4 Agents
1. Investigator — trained via GRPO, zero hardcoded financial knowledge
2. Criminal Designer — writes own Python attack code (Criminal Codex), morphs mid-episode
3. Compliance Expert — preference drift over episodes (Snorkel AI bonus)
4. Oversight Agent — monitors all agents in real time (Fleet AI bonus)

## The Transaction Network
- 100k nodes: accounts, shell companies, individuals, crypto exchanges
- 1M+ edges: transactions with amount, timestamp, type, jurisdiction
- Built using IBM AMLSim statistical properties
- 3-phase laundering: Placement → Layering → Integration
- Schema drift: new jurisdictions, thresholds, instruments between episode batches (Patronus AI bonus)

## The 6 Tools
query_transactions(entity_id, date_range, min_amount)
trace_network(seed_entity, max_depth, direction)
lookup_entity_info(entity_id)
cross_reference_jurisdiction(entity_id, country)
file_SAR(evidence_chain, risk_score, narrative)
request_subpoena(entity_id, justification)

## The 3 Key Innovations
1. Criminal Codex: criminal writes actual executable Python for new schemes, saved to criminal_codex.py. Episode 0: 4 schemes. Episode 50: 30+ schemes the criminal wrote itself. All schemes pass validation suite before entering Codex.
2. Mid-Episode Morphing: if investigator Bayesian belief on key entity exceeds P(criminal) > 0.7, criminal morphs — reroutes funds mid-episode, partially invalidates evidence chain. Extensive-form Stackelberg game. Maximum 1 morph per episode.
3. Zero-Day Reveal: after training, criminal generates best unconstrained scheme. Graph edit distance confirms novelty. Trained investigator catches it only 54%. Novel AI-invented crime pattern. Demo closing moment.

## The 4 Judge Defense Additions
A. Rule-based AML baseline: 50 lines Python. Proves trained agent (0.90 F1) crushes rules (0.30 F1) and untrained starts worse than rules (0.14 F1 — proves zero hardcoded knowledge).
B. Evidence chain visualization: each tool call lights up node/edge in yellow. Path grows visibly from dirty source to clean integration. Before/after: untrained dots scattered randomly, trained forms clean path.
C. Degenerate case defense: Criminal Codex validation suite, graph edit distance novelty measurement, causal reasoning explanation for morphing.
D. Why GRPO: no value function needed, PPO collapses on sparse 50-step episodes, DPO needs curated pairs which don't exist for novel adversarial schemes.

## Reward Functions
R_investigator = w1*detection_f1 + w2*evidence_quality + w3*query_efficiency + w4*jurisdiction_compliance - w5*false_positive_penalty + w6*novel_scheme_bonus - w7*missed_novel_penalty
R_criminal = laundering_volume × (1 - detection_rate) × novelty_bonus × morph_success_bonus
R_oversight = anomalies_caught / anomalies_total

## Advanced Math
1. Extensive-form Stackelberg Game (mid-episode morphing)
2. Bayesian Belief Updating on entity criminality
3. Information-theoretic query selection (mutual information maximization)
4. Red Queen Curriculum (softmax weakness targeting, temperature decay)
5. Suspicious Path Detection (learned Bellman-Ford on transaction graph)
6. Shapley Value Attribution
7. Graph Edit Distance (novelty measurement for Criminal Codex and Zero-Day)
8. Hindsight Experience Replay (Andrychowicz et al., 2017) — adapted for LLM tool-use agents with graph-structured virtual goals
9. Frequent Subsequence Mining for investigation skill discovery (Options Framework, Sutton et al., 1999)

## Themes Targeted
Theme 1: Multi-Agent (4 agents opposing reward functions)
Theme 2: Long Horizon (28-50 steps, 4 phases, evidence chain visualization proves it)
Theme 3.1: World Modeling Professional (IBM AMLSim graph, real tool calls)
Theme 4: Self-Improvement (criminal writes own attack code — Criminal Codex)

## Bonus Prizes
Fleet AI: Oversight Agent monitors all agents
Snorkel AI: Compliance Expert preference drift
Patronus AI: Regulatory schema drift
Halluminate: 4 distinct actors in multi-agent environment

## Build Order
Step 1: Transaction Graph
Step 2: OpenEnv Core
Step 3: Tool Execution Layer
Step 4: Wire tools.py into heist_env.py + inference.py
Step 5: Reward Calculator
Step 6: Scenario Library
Step 7: Criminal Designer + Criminal Codex + Validation Suite
Step 8: Mid-Episode Morphing
Step 9: Compliance Expert Agent
Step 10: Oversight Agent
Step 11: GRPO Training
Step 12: Red Queen Curriculum
Step 13: Evaluation + Zero-Day Reveal + Rule-Based Baseline + Graph Edit Distance
Step 14: Investigation War Room UI + Evidence Chain Visualization
Step 15: HuggingFace + Colab
Step 16: Final Polish + Demo Script + Judge Defense Q&A
Step 17: Hindsight Experience Replay (HER) — recycles failed episodes into training signal
Step 18: Investigation Skill Discovery — mines reusable patterns from successful episodes
