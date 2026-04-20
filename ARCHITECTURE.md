# HEIST — Architecture

## File Structure
heist/
  env/
    transaction_graph.py    — 100k node graph, IBM AMLSim properties, scheme injection
    heist_env.py            — OpenEnv interface, phase tracking, morph trigger
    models.py               — Pydantic Action/Observation/State models
    tools.py                — 6 tool implementations + Bayesian belief updater
    reward.py               — R_investigator + R_criminal + R_oversight
    scenarios.py            — 30 hand-crafted seed scenarios
    server/
      app.py                — create_app() factory (OpenEnv scaffold)
  agents/
    criminal.py             — Criminal Designer + Codex + validation suite + morphing
    expert.py               — Compliance Expert with preference drift
    oversight.py            — Fleet AI oversight agent
  train/
    train_grpo.py           — GRPO training with TRL + Unsloth
    curriculum.py           — Red Queen curriculum + ELO tracker
  eval/
    evaluate.py             — Evaluation + Zero-Day Reveal + rule baseline + edit distance
  ui/
    app.py                  — Streamlit Investigation War Room + evidence chain viz
criminal_codex.py           — AI-written criminal schemes (grows during training)
openenv.yaml                — spec_version: 1
pyproject.toml              — uv dependencies
heist_colab.ipynb           — Training demo notebook
README.md                   — Architecture + setup + judge defense Q&A

## Key Design Decisions
See DECISIONS.md
