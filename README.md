# HEIST 🔍: AI Detective vs AI Criminal

**A Co-Evolving Multi-Agent Environment for Money Laundering Detection**  
Built for the Meta Hackathon Finals.

> *"An AI detective and an AI criminal. Neither has hardcoded knowledge. The criminal writes its own attack code. Both co-evolve for 50 episodes. At the end, the criminal generates a novel money laundering pattern that no human designed — and our trained investigator still can’t fully catch it."*

---

## 🏛️ Architecture

```text
                                +-----------------------------------+
                                |         Transaction Graph         |
                                |   (100k nodes, IBM AMLSim base)   |
                                +-----------------------------------+
                                      ^                       ^
            +--------------------+    |                       |    +-------------------------+
            |  Investigator AI   |----|   [OpenEnv Interface] |----|    Criminal Designer    |
            |   (Qwen2.5-1.5B)   |    v                       v    |  (LLM writing Python)   |
            |                    |  [6 API Tools]       [Morphing] |                         |
            | GRPO Trained       |                                 | Updates Criminal Codex  |
            +--------------------+                                 +-------------------------+
                     |                                                          ^
                     |   +-------------------+       +--------------------+     |
                     +-->| Compliance Expert |<----->|  Oversight Agent   |<----+
                     |   | (Snorkel Drift)   |       | (Fleet AI Monitor) |     |
                     |   +-------------------+       +--------------------+     |
                     |             v                                            |
                     +------> [ Reward ] ---------------------------------------+
```

## 🚀 Features

1. **Criminal Codex**: The criminal agent writes real, executable Python code to generate new attack schemes during training, expanding its attack catalog dynamically.
2. **Mid-Episode Morphing**: Stackelberg game mechanics. If the investigator gets too close (P(criminal) > 0.7), the criminal morphs the scheme mid-episode, evading detection.
3. **Red Queen Curriculum**: Softmax weakness targeting. As the investigator improves, the criminal exponentially targets the scheme types where the investigator F1 score is lowest.
4. **Investigation War Room**: Streamlit-based UI with interactive pyvis graph exploration, evidence chain tracking, Bayesian belief bars, and zero-day testing.

---

## 🛠️ Setup Instructions

### 1. Install Dependencies
```bash
# We use uv for fast package management
pip install uv
uv venv
# Windows: .venv\Scripts\activate
# Mac/Linux: source .venv/bin/activate

uv run pip install -e ./env
```

### 2. Set API Keys
Export the following environment variables. The `GEMINI_API_KEY` is required for the Criminal Designer, Compliance Expert, and Oversight Agent.
```bash
# Windows
set GEMINI_API_KEY=your_gemini_api_key
set HF_TOKEN=your_huggingface_token_optional

# Linux/Mac
export GEMINI_API_KEY="your_gemini_api_key"
export HF_TOKEN="your_huggingface_token_optional"
```

### 3. Launch the War Room UI (HuggingFace Space)
```bash
uv run streamlit run ui/app.py
```
> The `app.py` in the root folder automatically redirects to `ui/app.py` to ensure HuggingFace Space auto-deployments work seamlessly.

---

## 🧪 Example Episode Output

```text
[PHASE:AlertTriage] [BUDGET:50]
FLAGGED: acc_209 -> shell_company_4922 $15400
TOP_SUSPECTS: none
MORPH_ALERT: none
CHOOSE ACTION: ACTION:query_transactions ENTITY:acc_209

  step= 1 | query_transactions | R=+0.04 | budget=49
  ...
  step= 6 | trace_network      | R=+0.12 | budget=44
  ...
[PHASE:Investigation] [BUDGET:44]
FLAGGED: acc_209 -> shell_company_4922 $15400
TOP_SUSPECTS: acc_209=0.85 | shell_company_4922=0.72
MORPH_ALERT invalidated=['shell_company_4922']
CHOOSE ACTION: ACTION:cross_reference_jurisdiction ENTITY:acc_209
  ...
[PHASE:SARFiling] [BUDGET:38]
...
CHOOSE ACTION: ACTION:file_SAR ENTITIES:acc_209,shell_company_8109,crypto_exchange_4
  step=12 | file_SAR       | R=+0.86 | budget=38
```

---

## 📜 Criminal Codex

The `criminal_codex.py` acts as the evolving brain of the AI criminal. 
- It starts with exactly **4 human-designed Seed Schemes** (e.g., `rapid_structuring`, `crypto_tumbler`).
- As training iterations proceed, the Criminal Designer LLM targets investigator weaknesses and dynamically *writes and appends* new python methods to the Codex (e.g., `inject_scheme_trade_1773()`).
- All code runs through a 5-check validation suite before making it into the Codex.

## 💣 Zero-Day Reveal
At the end of training, the AI criminal produces a **Zero-Day laundering pattern**. 
A structural Graph Edit Distance (GED) calculation confirms it is novel and never-before-seen. The heavily trained investigator AI agent, possessing a 0.90 F1 detection score on known schemes, catches the Zero-Day pattern at less than 30% F1, demonstrating the true innovation gap of the adversarial environment.
