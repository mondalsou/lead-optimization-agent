# ⚡ Quick Start — Lead Optimization Agent

Get running in under 5 minutes.

---

## Prerequisites

- Python 3.11+
- An Anthropic API key → [console.anthropic.com](https://console.anthropic.com) (free $5 credit on signup)

---

## Step 1 — Clone & Install

```bash
git clone https://github.com/mondalsou/lead-optimization-agent.git
cd lead-optimization-agent
pip install -r requirements.txt
```

> **RDKit trouble?** Use conda instead:
> ```bash
> conda install -c conda-forge rdkit
> ```

---

## Step 2 — Set your API key

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

---

## Step 3 — Choose how to run

### A) Streamlit UI  *(recommended for demo)*

```bash
streamlit run app.py
```

Open **http://localhost:8501** → select the preset → click **▶ Run Agent**

---

### B) Jupyter Notebooks  *(recommended for walkthrough)*

```bash
jupyter notebook notebooks/
```

Run in order:

| Notebook | What to do |
|---|---|
| `01_admet_tool.ipynb` | Run all cells — explores the API |
| `02_agent_loop.ipynb` | **Main notebook** — runs the agent |
| `03_visualization.ipynb` | Charts & molecule grid |

---

## What happens when you run

```
You enter:  Atenolol SMILES + "improve CNS penetration"
                │
                ▼
     Agent analyzes starting molecule
                │
     Proposes structural change (removes amide group)
                │
     Validates SMILES → Analyzes new ADMET scores
                │
     Compares: is it better? → Continue or stop
                │
                ▼
     Best candidate + chemical reasoning
```

---

## Expected output (Streamlit)

After ~3–5 minutes you'll see:

| Metric | Before | After |
|---|---|---|
| CNS MPO | 4.55 | **5.45** ↑ |
| BBB Probability | 0.60 | **0.90** ↑ |
| QED | 0.638 | **0.838** ↑ |
| Rotatable Bonds | 8 | **6** ↓ |

---

## Cost

~**$0.10–0.15** per full run. The $5 free credit covers ~35–50 runs.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| First API call hangs ~60s | Normal — Render free tier cold start. Wait it out. |
| `ModuleNotFoundError: rdkit` | `conda install -c conda-forge rdkit` |
| `AuthenticationError` | Check `echo $ANTHROPIC_API_KEY` — must start with `sk-ant-` |
| Streamlit port in use | `streamlit run app.py --server.port 8502` |
