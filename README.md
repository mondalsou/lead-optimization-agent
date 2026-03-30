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

An AI-assisted medicinal chemistry sandbox for exploring lead-optimization ideas in a visual, iterative workflow.

The app combines:
- an Anthropic-powered agent loop for proposing structural changes
- local RDKit-based property analysis for fast scoring
- a Streamlit UI for reviewing each attempt, change rationale, and property trajectory

[![Live Demo](https://img.shields.io/badge/Live%20Demo-HuggingFace%20Spaces-FFD21E?style=flat&logo=huggingface&logoColor=black)](https://huggingface.co/spaces/mondalsou/lead_optimization_agent)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Anthropic](https://img.shields.io/badge/Anthropic-Claude-191919?style=flat)](https://anthropic.com)
[![RDKit](https://img.shields.io/badge/RDKit-local%20analysis-0f766e?style=flat)](https://www.rdkit.org/)

**[Try the live demo on HuggingFace Spaces](https://huggingface.co/spaces/mondalsou/lead_optimization_agent)**

## Agentic workflow architecture

This project demonstrates an **AI-native scientific workflow** — a pattern increasingly central to computational pharmaceutical development. Rather than producing a single model output, the system implements a human-in-the-loop iterative loop where AI reasoning, deterministic scientific tools, and scientist judgment work together:

```
Scientist defines goal (brief)
         │
         ▼
  Agent proposes structural edit  ←──────────────────┐
         │                                            │
         ▼                                            │
  RDKit scores candidate locally                      │
  (QED, BBB, ADMET, solubility, SA)                  │
         │                                            │
         ▼                                            │
  Candidate tracked in journey log                   │
         │                                            │
         ▼                                            │
  Scientist reviews: accept / redirect / stop ────────┘
         │
         ▼
  Best candidate surfaced with full audit trail
```

This architecture maps directly to how AI-assisted workflows are being embedded in drug product development: the model proposes, the deterministic tools validate, and the human expert retains oversight and final judgment. The agent does not make unilateral decisions — it accelerates the scientist's reasoning cycle.

**Why this matters for CMC/formulation contexts**: The same loop structure — propose formulation change → score against CQAs → refine with expert input → log decision rationale — applies directly to AI-supported formulation development and process optimization. The pattern here is domain-agnostic; the chemistry scoring layer is the swappable component.

---

## What this project does

Given a starting molecule and a target optimization brief, the agent:

1. validates the starting SMILES
2. analyzes the molecule locally with RDKit-derived heuristics
3. proposes one structural change at a time
4. scores each new candidate
5. compares property movement across attempts
6. surfaces the best candidate found in the run

The UI is built to answer the questions a chemist actually cares about:
- What changed in this attempt?
- Why was that change made?
- Did BBB, CNS MPO, QED, or flexibility improve?
- Which candidate is currently the best balance?

## Current UI highlights

The Streamlit app includes:
- `Candidate Journey` as the first tab, with live attempt-by-attempt cards
- highlighted 2D structures showing the region changed in each attempt
- short plain-English change summaries for each analogue
- `Performance Overview` with metric deltas, trajectory plots, and a start-vs-best radar chart
- local run persistence, so you can reload past runs without spending LLM credits again

## Example scenarios

The app ships with a few preset briefs:
- `Atenolol → Brain Penetration`
- `Aspirin → CNS Drug Profile`
- `Ibuprofen → Aqueous Solubility`
- `Custom molecule`

## How it works

### Agent loop

The agent is responsible for proposing the next chemical edit and explaining the logic behind it.

### Local chemistry scoring

Property analysis is performed locally in `agent_utils.py` using RDKit-based calculations and heuristics, including:
- QED
- Lipinski summary
- BBB probability heuristic
- CNS MPO heuristic
- solubility estimate
- GI absorption heuristic
- structural alerts
- synthetic accessibility heuristic

This means the chemistry scoring path is local and fast. The Anthropic API is used for the reasoning loop, not for property calculation.

## Project structure

```text
lead_optimization_agent/
├── app.py
├── agent_utils.py
├── requirements.txt
├── candidates.json
├── saved_runs/
└── notebooks/
    ├── 01_admet_tool.ipynb
    ├── 02_agent_loop.ipynb
    └── 03_visualization.ipynb
```

## Quick start

### 1. Clone and install

```bash
git clone https://github.com/mondalsou/lead_optimization_agent.git
cd lead_optimization_agent
pip install -r requirements.txt
```

If RDKit installation fails via `pip`, use:

```bash
conda install -c conda-forge rdkit
```

### 2. Set your Anthropic API key

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

### 3. Run the app

```bash
streamlit run app.py
```

Open `http://localhost:8501`.

## Using the app

1. Pick a preset or paste your own SMILES.
2. Write the optimization brief.
3. Click `Run Optimisation`.
4. Watch `Candidate Journey` update as new molecules are analyzed.
5. Open `Performance Overview` to compare start vs best candidate.

## Saved runs

Every completed run is saved locally so you can reopen it later without calling the API again.

Saved files are written to:

```text
saved_runs/latest_run.json
saved_runs/run_YYYYMMDD_HHMMSS.json
```

From the sidebar you can:
- load the latest saved run
- upload a saved JSON run
- download the current run

## Notebooks

The notebooks are still useful for exploration and demos:
- `01_admet_tool.ipynb`: property exploration and tool setup
- `02_agent_loop.ipynb`: agent-loop walkthrough
- `03_visualization.ipynb`: charts and candidate visualization

## Requirements

Core dependencies:

```text
streamlit>=1.35.0
anthropic>=0.40.0
rdkit>=2023.9.5
pandas>=2.0.0
plotly>=5.18.0
matplotlib>=3.8.0
```

## What this repo is good for

- demonstrating AI-native scientific workflows with human-in-the-loop oversight
- showing iterative agent behavior and decision tracking instead of one-shot prompting
- medicinal chemistry and formulation workflow prototyping
- illustrating how LLM reasoning and deterministic scientific tools can be composed
- experimenting with optimization briefs and visual candidate review

## Limitations

- this is a heuristic prototyping tool, not a validated drug-discovery platform
- property outputs are local approximations and should not be treated as experimental truth
- agent suggestions should be reviewed by a domain expert before any serious decision-making

## Why this is interesting

This project sits at the intersection of:
- agentic workflows for scientific decision support
- chemistry-aware UI design for expert stakeholders
- human-in-the-loop optimization with full audit trail
- LLM reasoning paired with deterministic, reproducible local analysis
- decision-support tooling that keeps scientific oversight central — not a black box

## Author

Sourav Mondal
- GitHub: [@mondalsou](https://github.com/mondalsou)
- LinkedIn: [Sourav Mondal](https://www.linkedin.com/in/soura1/)
