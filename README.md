---
title: Lead Optimization Agent
emoji: 🧬
colorFrom: green
colorTo: blue
sdk: streamlit
sdk_version: 1.35.0
app_file: app.py
pinned: false
short_description: AI agent for iterative drug lead optimization with RDKit
---

# Lead Optimization Agent

[![Live Demo](https://img.shields.io/badge/Live%20Demo-HuggingFace%20Spaces-FFD21E?style=flat&logo=huggingface&logoColor=black)](https://huggingface.co/spaces/mondalsou/lead_optimization_agent)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Anthropic](https://img.shields.io/badge/Anthropic-Claude-191919?style=flat)](https://anthropic.com)
[![RDKit](https://img.shields.io/badge/RDKit-local%20analysis-0f766e?style=flat)](https://www.rdkit.org/)

**[Try the live demo on HuggingFace Spaces](https://huggingface.co/spaces/mondalsou/lead_optimization_agent)**

---

## Problem

Drug lead optimization is one of the most expensive and slow phases of drug discovery. A medicinal chemist starts with a promising molecule and must iteratively explore chemical space — proposing analogues, synthesizing them, measuring ADMET properties (absorption, distribution, metabolism, excretion, toxicity), and deciding whether to continue or pivot.

This process is:
- **Manual and slow** — each analogue requires expert chemical intuition to design
- **Disconnected** — scoring tools, structure editors, and decision notes live in separate places
- **Opaque** — it's hard to audit why a particular direction was pursued across dozens of iterations
- **Expensive to iterate** — without fast local scoring, every feedback loop requires wet-lab or commercial prediction services

There is no lightweight tool that closes the loop between AI-assisted structural reasoning, fast local property scoring, and scientist oversight — all in one interface.

---

## Solution

**Lead Optimization Agent** is an AI-native iterative sandbox for medicinal chemistry exploration. Given a starting molecule and a plain-English optimization goal, the agent proposes one structural change at a time, scores each candidate locally using RDKit, and surfaces results for the scientist to review, accept, or redirect.

The scientist stays in control. The agent accelerates the ideation and scoring cycle.

---

## How It Works

### Agentic loop

```
Scientist defines goal (brief)
         │
         ▼
  Agent proposes structural edit  ←──────────────────┐
         │                                            │
         ▼                                            │
  RDKit scores candidate locally                      │
  (QED, BBB, CNS MPO, solubility, SA score)          │
         │                                            │
         ▼                                            │
  Attempt logged with rationale + property delta      │
         │                                            │
         ▼                                            │
  Scientist reviews: accept / redirect / stop ────────┘
         │
         ▼
  Best candidate surfaced with full audit trail
```

### Two distinct components

**Agent (Claude via Anthropic API)** — handles the chemistry reasoning layer:
- reads the current molecule and property scores
- proposes the next structural edit and explains the logic
- does not perform property calculations

**Local scoring (RDKit in `agent_utils.py`)** — deterministic, fast, no API call:
- QED (drug-likeness)
- Lipinski rule-of-five summary
- BBB permeability heuristic
- CNS MPO score
- Aqueous solubility estimate
- GI absorption heuristic
- Structural alerts
- Synthetic accessibility (SA) heuristic

This separation means chemistry scoring is reproducible and free; the LLM is used only for reasoning about what change to try next.

### UI

The Streamlit app presents each iteration as a candidate card with:
- 2D structure with the changed region highlighted
- plain-English change summary
- per-property deltas vs. the starting molecule
- `Candidate Journey` tab for attempt-by-attempt review
- `Performance Overview` tab with trajectory plots and a start-vs-best radar chart

Each completed run is saved automatically to `saved_runs/YYYY-MM-DD/run_HHMMSS.json` and can be reloaded from the sidebar without calling the API again. `latest_run.json` is always kept at the root of `saved_runs/` and auto-loaded on startup.

---

## Quick Start

```bash
git clone https://github.com/mondalsou/lead_optimization_agent.git
cd lead_optimization_agent
pip install -r requirements.txt
# if RDKit fails via pip: conda install -c conda-forge rdkit
export ANTHROPIC_API_KEY=sk-ant-...
streamlit run app.py
```

Open `http://localhost:8501`, pick a preset or paste your own SMILES, write the optimization brief, and click **Run Optimisation**.

---

## Project Structure

```
lead_optimization_agent/
├── app.py              # Streamlit UI + agent orchestration
├── agent_utils.py      # RDKit scoring, SMILES validation, helpers
├── requirements.txt
├── saved_runs/         # Run persistence — one subfolder per day
│   ├── latest_run.json          # Always the most recent run (auto-loaded on startup)
│   └── YYYY-MM-DD/
│       └── run_HHMMSS.json      # Timestamped run snapshot
└── notebooks/
    ├── 01_admet_tool.ipynb
    ├── 02_agent_loop.ipynb
    └── 03_visualization.ipynb
```

---

## Preset Scenarios

- `Atenolol → Brain Penetration`
- `Aspirin → CNS Drug Profile`
- `Ibuprofen → Aqueous Solubility`
- `Custom molecule`

---

## Limitations

- Heuristic prototyping tool, not a validated drug-discovery platform
- Property outputs are local approximations, not experimental measurements
- Agent suggestions should be reviewed by a domain expert before serious decisions

---

## Author

Sourav Mondal — [GitHub](https://github.com/mondalsou) · [LinkedIn](https://www.linkedin.com/in/soura1/)
