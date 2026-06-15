---
title: Lead Optimization Agent
emoji: 🧬
colorFrom: green
colorTo: blue
sdk: streamlit
sdk_version: 1.35.0
app_file: app.py
pinned: false
short_description: LangChain agent for iterative drug lead optimization with RDKit
---

# Lead Optimization Agent

[![Live Demo](https://img.shields.io/badge/Live%20Demo-HuggingFace%20Spaces-FFD21E?style=flat&logo=huggingface&logoColor=black)](https://huggingface.co/spaces/mondalsou/lead_optimization_agent)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io)
[![LangChain](https://img.shields.io/badge/LangChain-0.2+-1C3C3C?style=flat&logo=langchain&logoColor=white)](https://langchain.com)
[![OpenRouter](https://img.shields.io/badge/OpenRouter-multi--model-6366f1?style=flat)](https://openrouter.ai)
[![RDKit](https://img.shields.io/badge/RDKit-local%20scoring-0f766e?style=flat)](https://www.rdkit.org/)

**[Try the live demo on HuggingFace Spaces](https://huggingface.co/spaces/mondalsou/lead_optimization_agent)**

---

## What it does

LangChain agent that iteratively optimizes drug molecules toward a target ADMET profile. Give it a starting SMILES and a plain-English goal — it proposes one structural change per round, scores each candidate locally via RDKit, and surfaces results in a Streamlit UI.

Swap between Claude, GPT-4o, DeepSeek V4 Flash, Gemini, or Llama by changing one dropdown. No model-specific code.

---

## Problem

Drug lead optimization is slow and expensive. A medicinal chemist starts with a promising molecule and must iteratively explore chemical space — proposing analogues, scoring ADMET properties, and deciding whether to continue or pivot. Each feedback loop requires expert chemical intuition and either wet-lab measurements or commercial prediction services.

This project closes that loop: AI reasoning + fast local scoring + scientist oversight, in one interface.

---

## How It Works

### Agentic loop

```
Scientist defines goal (SMILES + brief)
         │
         ▼
  Agent proposes structural edit  ←──────────────────┐
         │                                            │
         ▼                                            │
  RDKit scores candidate locally                      │
  (QED, BBB, CNS MPO, solubility, alerts)            │
         │                                            │
         ▼                                            │
  Attempt logged with rationale + property delta      │
         │                                            │
         ▼                                            │
  Scientist reviews → accept / redirect / stop ───────┘
         │
         ▼
  Best candidate surfaced with full audit trail
```

### Stack

| Layer | What |
|---|---|
| **LLM routing** | LangChain + OpenRouter — any tool-calling model |
| **Agent loop** | `create_tool_calling_agent` + `AgentExecutor` |
| **Chemistry scoring** | RDKit (local, instant, free) via `agent_utils.py` |
| **UI** | Streamlit with live incremental updates |

### Tools the agent has

| Tool | What it does |
|---|---|
| `validate_smiles` | RDKit SMILES validation before analysis |
| `analyze_molecule` | Full ADMET profile — QED, BBB, CNS MPO, solubility, Lipinski, alerts |
| `compare_candidates` | Side-by-side scoring of multiple molecules |

Scoring is deterministic and local — no API call for chemistry. The LLM is used only for structural reasoning.

---

## Quick Start

```bash
git clone https://github.com/mondalsou/lead-optimization-agent.git
cd lead-optimization-agent
pip install -r requirements.txt
# RDKit via pip may fail — use conda if so:
# conda install -c conda-forge rdkit

cp .env.example .env
# add your OpenRouter key to .env:
# OPENROUTER_API_KEY=sk-or-...

streamlit run app.py
```

Open `http://localhost:8501`, pick a preset or paste your own SMILES, write the optimization brief, pick a model, and click **Run Optimisation**.

---

## Supported Models (via OpenRouter)

| Model | Slug |
|---|---|
| Claude Sonnet 4.6 | `anthropic/claude-sonnet-4.6` |
| Claude Haiku 4.5 | `anthropic/claude-haiku-4.5` |
| Claude Opus 4.6 | `anthropic/claude-opus-4.6` |
| DeepSeek V4 Flash | `deepseek/deepseek-v4-flash` |
| GPT-4o | `openai/gpt-4o` |
| Gemini 2.0 Flash | `google/gemini-2.0-flash-001` |
| Llama 3.3 70B | `meta-llama/llama-3.3-70b-instruct` |

---

## Project Structure

```
lead-optimization-agent/
├── app.py                  # Streamlit UI + LangChain agent
├── agent_utils.py          # RDKit scoring, SMILES validation
├── requirements.txt
├── .env.example            # Copy to .env and add your key
├── saved_runs/             # Local run persistence (JSON)
├── hermes_tools/           # Hermes agent integration
│   ├── lead_opt.py
│   └── chemistry_skill.md
├── hermes_setup.sh         # One-shot Hermes setup script
└── notebooks/
    ├── 01_admet_tool.ipynb         # Explore the ADMET scorer
    ├── 02_agent_loop.ipynb         # Anthropic SDK agent loop
    ├── 03_visualization.ipynb      # Optimization trajectory plots
    └── 04_langchain_openrouter.ipynb  # LangChain + OpenRouter version
```

---

## Preset Scenarios

- `Atenolol → Brain Penetration` — reduce polarity, improve BBB / CNS MPO
- `Aspirin → CNS Drug Profile` — replace carboxylic acid liability
- `Ibuprofen → Aqueous Solubility` — lower cLogP, add polar groups
- `Custom molecule` — paste any SMILES

---

## Limitations

- Heuristic prototyping tool, not a validated drug-discovery platform
- Property outputs are local approximations, not experimental measurements
- Agent suggestions should be reviewed by a domain expert

---

## Author

Sourav Mondal — [GitHub](https://github.com/mondalsou) · [LinkedIn](https://www.linkedin.com/in/soura1/)
