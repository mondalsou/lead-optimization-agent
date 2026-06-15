#!/usr/bin/env bash
# hermes_setup.sh — Install lead-optimization-agent tools into Hermes
# Run once on the VM after cloning the repo:
#   git clone <repo> && cd lead-optimization-agent && bash hermes_setup.sh

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Auto-detect Hermes installation
_HERMES_BASE=""
for _candidate in /usr/local/lib/hermes-agent "$HOME/.hermes" /opt/hermes-agent; do
    if [ -f "$_candidate/tools/registry.py" ]; then
        _HERMES_BASE="$_candidate"
        break
    fi
done

HERMES_TOOLS_DIR="${HERMES_TOOLS_DIR:-${_HERMES_BASE:+$_HERMES_BASE/tools}}"
HERMES_SKILLS_DIR="${HERMES_SKILLS_DIR:-${_HERMES_BASE:+$_HERMES_BASE/skills}}"
HERMES_TOOLS_DIR="${HERMES_TOOLS_DIR:-$HOME/.hermes/tools}"
HERMES_SKILLS_DIR="${HERMES_SKILLS_DIR:-$HOME/.hermes/skills}"

# Prefer Hermes venv Python/pip if present
if [ -n "$_HERMES_BASE" ] && [ -f "$_HERMES_BASE/venv/bin/pip" ]; then
    PIP="$_HERMES_BASE/venv/bin/pip"
    PYTHON="$_HERMES_BASE/venv/bin/python"
    PIP_FLAGS=""
    echo "  Using Hermes venv: $_HERMES_BASE/venv"
else
    # Fall back to system pip with PEP 668 handling
    if command -v pip3 &>/dev/null; then PIP=pip3; elif command -v pip &>/dev/null; then PIP=pip; else PIP="python3 -m pip"; fi
    if command -v python3 &>/dev/null; then PYTHON=python3; else PYTHON=python; fi
    PIP_FLAGS=""
    _pip_test=$($PIP install --dry-run pip 2>&1 || true)
    if echo "$_pip_test" | grep -q "externally-managed"; then
        PIP_FLAGS="--break-system-packages"
        echo "  Note: system Python is externally managed, using --break-system-packages"
    fi
fi

echo "=== Lead Optimization Agent — Hermes Setup ==="
echo "Repo:         $REPO_DIR"
echo "Hermes tools: $HERMES_TOOLS_DIR"
echo "Hermes skills: $HERMES_SKILLS_DIR"
echo

# ── 1. Install Python dependencies ─────────────────────────────────────────
echo "[1/4] Installing Python dependencies..."
$PIP install --quiet $PIP_FLAGS streamlit anthropic pandas plotly matplotlib 2>&1 | tail -1

# ── 2. Install RDKit ────────────────────────────────────────────────────────
echo "[2/4] Installing RDKit..."
if $PYTHON -c "import rdkit" 2>/dev/null; then
    echo "  RDKit already installed."
else
    if $PIP install --quiet $PIP_FLAGS rdkit 2>/dev/null; then
        $PYTHON -c "import rdkit" && echo "  RDKit installed via pip." || {
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
            echo "  ERROR: Cannot install RDKit (no conda, $PIP failed)."
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
