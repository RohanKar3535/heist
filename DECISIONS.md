# HEIST — Key Decisions (DO NOT REVERSE)

D1: networkx graph not real database
Reason: Runs on Colab and HuggingFace Spaces. Resets in milliseconds. IBM AMLSim citation gives credibility equal to real data.

D2: Gemini API for Criminal/Expert/Oversight LLM calls
Reason: Free tier sufficient. Qwen2.5-1.5B for GRPO training fits Colab T4.

D3: Criminal writes actual Python code (Criminal Codex)
Reason: Theme 4 self-improvement becomes recursive — AI programming new attacks against AI defense. Demo moment: show file growing from 4 to 31 schemes. All schemes pass validation suite before entering Codex.

D4: Mid-episode morphing triggers at P(criminal) > 0.7, maximum 1 morph per episode
Reason: Upgrades standard Stackelberg to extensive-form game. Maximum 1 morph prevents reward signal noise. Investigator learns to commit early.

D5: Softmax weakness targeting with temperature decay
Reason: Mathematical guarantee investigator cannot overfit to any scheme type.

D6: Shapley values for compliance expert attribution
Reason: Fair credit in multi-agent setting. Judges understand cooperative game theory.

D7: 4 phases per episode
Reason: Forces genuine long-horizon reasoning. Evidence chain visualization makes this visible to judges.

D8: Oversight agent has own reward function
Reason: Makes Fleet AI bonus prize claim legitimate.

D9: Schema drift between episode batches not mid-episode
Reason: Mid-episode drift makes reward calculation ambiguous. Batch drift satisfies Patronus AI bonus cleanly.

D10: Zero-Day Reveal as demo closing moment
Reason: AI-invented crime pattern. Graph edit distance confirms novelty. Unforgettable closing line.

D11: OpenEnv latest scaffold (pyproject.toml + uv)
Reason: openenv validate fails with requirements.txt/pip.

D12: HEIST not PROMETHEUS
Reason: Domain novelty. Financial crime narrative. Host confirmed any PS allowed. "AI detective vs AI criminal" story lands harder than infrastructure.

D13: Rule-based AML baseline
Reason: Proves untrained agent starts worse than rules (zero knowledge). Trained agent crushes rules. One chart tells entire story to judges.

D14: Evidence chain visualization
Reason: Makes Theme 2 long-horizon planning visible. Before/after is devastating. Untrained = scattered dots. Trained = clean path from source to sink.

D15: Why GRPO not PPO or DPO
Reason: No value function needed. PPO collapses on sparse 50-step episodes. DPO needs curated pairs which don't exist for novel adversarial schemes.

D16: Hindsight Experience Replay (HER) for sparse reward
Reason: Most episodes fail (F1 < 0.5). Standard GRPO wastes these — near-zero gradient signal. HER relabels the ground truth to match the agent's actual investigation path, converting every failed episode into training signal. Novel application: HER has never been applied to LLM tool-use agents. Virtual goals must form connected subgraphs (not just random entity sets) — this is the key adaptation for graph-structured environments.

D17: Investigation Skill Discovery via frequent subsequence mining
Reason: Successful investigations share common action patterns (e.g., "query → trace → trace → cross-ref"). Mining these patterns and injecting them into the system prompt as advisory strategies gives the agent reusable investigation playbooks. Reusability angle: other tool-augmented RL projects can import HEIST's discovered skills library. Based on Options Framework (Sutton et al., 1999).
