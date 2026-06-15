"""
Hermes tool wrappers for the Lead Optimization Agent.

Drop this file (or symlink it) into your Hermes tools/ directory.
Hermes auto-discovers it at startup via registry.register() calls.

Setup: run hermes_setup.sh from the repo root on your VM.
"""
import json
import os
import sys

# Make agent_utils importable regardless of where Hermes loads this from
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

try:
    from agent_utils import tool_executor
    _RDKIT_OK = True
except ImportError as e:
    _RDKIT_OK = False
    _IMPORT_ERROR = str(e)

from hermes.tools.registry import registry


def _rdkit_guard():
    if not _RDKIT_OK:
        return {"error": f"RDKit not available: {_IMPORT_ERROR}. Run: pip install rdkit"}
    return None


@registry.register(
    name="validate_smiles",
    description=(
        "Validate a SMILES string and return its canonical form. "
        "Always call this before analyze_molecule to catch typos."
    ),
    parameters={
        "type": "object",
        "properties": {
            "smiles": {
                "type": "string",
                "description": "SMILES string to validate",
            }
        },
        "required": ["smiles"],
    },
)
def validate_smiles(smiles: str) -> dict:
    err = _rdkit_guard()
    if err:
        return err
    return tool_executor("validate_smiles", {"smiles": smiles})


@registry.register(
    name="analyze_molecule",
    description=(
        "Compute a full ADMET profile for a molecule (local RDKit, no API). "
        "Returns MW, cLogP, TPSA, QED, BBB penetration, CNS MPO, solubility, "
        "Lipinski rules, PAINS alerts, synthetic accessibility, and decision guidance."
    ),
    parameters={
        "type": "object",
        "properties": {
            "smiles": {
                "type": "string",
                "description": "Valid canonical SMILES (validate first)",
            }
        },
        "required": ["smiles"],
    },
)
def analyze_molecule(smiles: str) -> dict:
    err = _rdkit_guard()
    if err:
        return err
    return tool_executor("analyze_molecule", {"smiles": smiles})


@registry.register(
    name="compare_candidates",
    description=(
        "Compare multiple drug candidates side by side. "
        "Returns ADMET scores for all molecules in one call."
    ),
    parameters={
        "type": "object",
        "properties": {
            "smiles_list": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of SMILES strings to compare",
            },
            "labels": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional display names for each molecule",
            },
        },
        "required": ["smiles_list"],
    },
)
def compare_candidates(smiles_list: list, labels: list = None) -> dict:
    err = _rdkit_guard()
    if err:
        return err
    inp = {"smiles_list": smiles_list}
    if labels:
        inp["labels"] = labels
    return tool_executor("compare_candidates", inp)
