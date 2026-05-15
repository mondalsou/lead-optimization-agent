#!/usr/bin/env bash
# hermes_setup.sh — Install lead-optimization-agent tools into Hermes
# Run once on the VM after cloning the repo:
#   git clone <repo> && cd lead-optimization-agent && bash hermes_setup.sh

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HERMES_TOOLS_DIR="${HERMES_TOOLS_DIR:-$HOME/.hermes/tools}"
HERMES_SKILLS_DIR="${HERMES_SKILLS_DIR:-$HOME/.hermes/skills}"

echo "=== Lead Optimization Agent — Hermes Setup ==="
echo "Repo:         $REPO_DIR"
echo "Hermes tools: $HERMES_TOOLS_DIR"
echo "Hermes skills: $HERMES_SKILLS_DIR"
echo

# ── 1. Install Python dependencies ─────────────────────────────────────────
echo "[1/4] Installing Python dependencies..."
pip install --quiet streamlit anthropic pandas plotly matplotlib 2>&1 | tail -1

# ── 2. Install RDKit ────────────────────────────────────────────────────────
echo "[2/4] Installing RDKit..."
if python -c "import rdkit" 2>/dev/null; then
    echo "  RDKit already installed."
else
    if pip install --quiet rdkit 2>/dev/null; then
        python -c "import rdkit" && echo "  RDKit installed via pip." || {
            echo "  pip install succeeded but import failed — trying conda..."
            if command -v conda &>/dev/null; then
                conda install -y -c conda-forge rdkit -q
            else
                echo "  ERROR: RDKit import failed and conda not available."
                echo "  Fix: conda install -c conda-forge rdkit  OR  pip install rdkit"
                exit 1
            fi
        }
    else
        echo "  pip install failed — trying conda..."
        if command -v conda &>/dev/null; then
            conda install -y -c conda-forge rdkit -q
        else
            echo "  ERROR: Cannot install RDKit (no conda, pip failed)."
            echo "  Fix: conda install -c conda-forge rdkit"
            exit 1
        fi
    fi
fi

# ── 3. Symlink tool file into Hermes tools/ ────────────────────────────────
echo "[3/4] Linking tool file..."
if [ ! -d "$HERMES_TOOLS_DIR" ]; then
    echo "  ERROR: Hermes tools dir not found: $HERMES_TOOLS_DIR"
    echo "  Check your Hermes install. Override with HERMES_TOOLS_DIR=/path/to/tools bash hermes_setup.sh"
    exit 1
fi
ln -sf "$REPO_DIR/hermes_tools/lead_opt.py" "$HERMES_TOOLS_DIR/lead_opt.py"
echo "  Linked: $HERMES_TOOLS_DIR/lead_opt.py -> $REPO_DIR/hermes_tools/lead_opt.py"

# ── 4. Install chemistry skill ─────────────────────────────────────────────
echo "[4/4] Installing chemistry skill..."
if [ -d "$HERMES_SKILLS_DIR" ]; then
    SKILL_DIR="$HERMES_SKILLS_DIR/lead-optimization"
    mkdir -p "$SKILL_DIR"
    cp "$REPO_DIR/hermes_tools/chemistry_skill.md" "$SKILL_DIR/SKILL.md"
    echo "  Skill installed: $SKILL_DIR/SKILL.md"
else
    echo "  Skipping skill install (no skills dir at $HERMES_SKILLS_DIR)."
fi

# ── Done ───────────────────────────────────────────────────────────────────
echo
echo "=== Setup complete ==="
echo "Restart Hermes to load the tools, then test via Telegram:"
echo '  validate the SMILES CC(C)NCC(O)COc1ccc(CC(N)=O)cc1'
echo '  analyze the molecule CC(=O)Oc1ccccc1C(=O)O'
