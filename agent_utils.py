"""
Lead Optimization Agent — Utility Functions
============================================
Thin wrappers around RDKit, exposing ADMET analysis as tool-callable
functions for the Claude agent. Fully local — no external API required.
"""
import math
from typing import Optional

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors, QED, Crippen
    from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("Warning: RDKit not available. Install with: conda install -c conda-forge rdkit")


# ─────────────────────────────────────────────────────────────────────────────
# Internal: Local ADMET Computation (pure RDKit, no network)
# ─────────────────────────────────────────────────────────────────────────────

def _des(val: float, lo: float, hi: float, lo_d: float = 1.0, hi_d: float = 0.0) -> float:
    """Linear desirability: clamps to [lo_d, hi_d] outside [lo, hi] range."""
    if lo_d >= hi_d:  # decreasing: higher val → lower desirability
        if val <= lo:
            return lo_d
        if val >= hi:
            return hi_d
    else:              # increasing: higher val → higher desirability
        if val <= lo:
            return lo_d
        if val >= hi:
            return hi_d
    t = (val - lo) / (hi - lo)
    return lo_d + t * (hi_d - lo_d)


def _has_smarts(mol, smarts: str) -> bool:
    try:
        pat = Chem.MolFromSmarts(smarts)
        return pat is not None and mol.HasSubstructMatch(pat)
    except Exception:
        return False



def _cyp_flags(mol) -> dict:
    """Heuristic CYP450 substrate/inhibitor flags using SMARTS."""
    # CYP3A4 substrate: large + aromatic + heteroatom (most drugs)
    mw = Descriptors.ExactMolWt(mol)
    has_ar = _has_smarts(mol, "a")
    has_n_or_o = _has_smarts(mol, "[N,O]")
    cyp3a4 = mw > 300 and has_ar and has_n_or_o

    # CYP2D6 substrate: basic N within 5 bonds of aromatic ring
    cyp2d6 = _has_smarts(mol, "[NX3;H0,H1,H2;!$(NC=O)]~*~*~a") or \
              _has_smarts(mol, "[NX3;H0,H1,H2;!$(NC=O)]~*~a")

    # Inhibitor risk: imidazole, pyridine N, or extended pi system
    cyp_inhib = (
        _has_smarts(mol, "n1ccnc1")   # imidazole-like
        or _has_smarts(mol, "n1ccccc1")  # pyridine
        or _has_smarts(mol, "c1ccc2ccccc2c1")  # naphthalene / fused rings
    )

    return {
        "cyp3a4_substrate": cyp3a4,
        "cyp2d6_substrate": cyp2d6,
        "cyp2c9_substrate": has_ar and _has_smarts(mol, "C(=O)[OH]"),  # carboxylic acid + ring
        "cyp_inhibitor_risk": cyp_inhib,
    }


def _esol_log_s(mol, mw: float, clogp: float, rb: int) -> float:
    """
    ESOL (Delaney 2004) solubility estimate:
    logS = 0.16 − 0.63·cLogP − 0.0062·MW + 0.066·RB − 0.74·AP
    AP = fraction aromatic carbons (proxy for aromaticity penalty)
    """
    ap = sum(1 for a in mol.GetAtoms()
             if a.GetIsAromatic() and a.GetAtomicNum() == 6) / max(mol.GetNumHeavyAtoms(), 1)
    return 0.16 - 0.63 * clogp - 0.0062 * mw + 0.066 * rb - 0.74 * ap


def _solubility_class(log_s: float) -> str:
    if log_s >= -1:   return "Highly Soluble"
    if log_s >= -2:   return "Soluble"
    if log_s >= -4:   return "Moderately Soluble"
    if log_s >= -6:   return "Poorly Soluble"
    return "Insoluble"


def _gi_absorption(tpsa: float, rb: int, mw: float, clogp: float, hbd: int) -> dict:
    """
    GI absorption: High if Veber rules pass AND Lipinski-like.
    Veber (2002): TPSA ≤ 140 AND rotatable bonds ≤ 10
    Also penalise if multiple Lipinski violations.
    """
    veber = tpsa <= 140 and rb <= 10
    lipinski_ok = mw <= 500 and clogp <= 5 and hbd <= 5
    if veber and lipinski_ok:
        absorption = "High"
        score = 0.85 + 0.15 * max(0, (140 - tpsa) / 140)
    elif veber or lipinski_ok:
        absorption = "Moderate"
        score = 0.55
    else:
        absorption = "Low"
        score = 0.20
    return {"absorption": absorption, "bioavailability_score": round(score, 2)}


def _bbb_probability(clogp: float, mw: float, tpsa: float, hbd: int) -> dict:
    """
    BBB penetration probability from rule-based weighted desirability.
    Reference ranges: Pajouhesh & Lenz (2005), Wager CNS MPO.
    """
    # Each parameter contributes 0-1 desirability
    d_logp = _des(clogp, 0.0, 5.0, 0.1, 1.0)   # cLogP 0-5 (peak ~2-3)
    if clogp > 3:
        d_logp = _des(clogp, 3.0, 5.0, 1.0, 0.3)

    d_mw   = _des(mw,   200.0, 450.0, 0.7, 0.0)  # MW < 400 strongly preferred
    d_tpsa = _des(tpsa, 40.0,  90.0,  1.0, 0.0)  # TPSA < 60 ideal
    d_hbd  = _des(float(hbd), 0.0, 3.0, 1.0, 0.0)

    # Weighted average (TPSA and HBD are most important for BBB)
    prob = 0.20 * d_logp + 0.20 * d_mw + 0.35 * d_tpsa + 0.25 * d_hbd
    prob = round(max(0.05, min(0.98, prob)), 3)
    penetrates = prob >= 0.5

    confidence = "High" if prob > 0.75 or prob < 0.25 else "Moderate"
    return {"penetrates": penetrates, "probability": prob, "confidence": confidence}


def _cns_mpo(clogp: float, mw: float, tpsa: float, hbd: int) -> dict:
    """
    CNS MPO score — Wager et al. 2010 (Pfizer), 5-parameter version.
    Each parameter contributes 0-1 desirability, summed to max 5.
    (pKa excluded — cannot be reliably computed from SMILES alone.)
    """
    d1 = _des(clogp,       3.0, 5.0, 1.0, 0.0)   # cLogP ≤ 3 → 1, ≥ 5 → 0
    d2 = _des(clogp,       1.0, 3.0, 0.0, 1.0)   # cLogD ≈ cLogP (neutral approx)
    if clogp > 3.0:
        d2 = _des(clogp,   3.0, 5.0, 1.0, 0.0)
    d3 = _des(mw,        360.0, 500.0, 1.0, 0.0)  # MW ≤ 360 → 1, ≥ 500 → 0
    d4 = _des(tpsa,       40.0,  90.0, 1.0, 0.0)  # TPSA ≤ 40 → 1, ≥ 90 → 0
    d5 = _des(float(hbd),  0.5,   3.5, 1.0, 0.0)  # HBD ≤ 0.5 → 1, ≥ 3.5 → 0

    score = round(d1 + d2 + d3 + d4 + d5, 2)

    if score >= 4.0:
        cns_class = "CNS+"
    elif score >= 3.0:
        cns_class = "CNS Borderline"
    else:
        cns_class = "CNS-"

    return {"score": score, "cns_class": cns_class}


def _lipinski_check(mw: float, clogp: float, hbd: int, hba: int) -> tuple:
    """Returns (rules list, summary string)."""
    rules = [
        {"name": "MW ≤ 500",        "value": mw,    "pass": mw    <= 500},
        {"name": "cLogP ≤ 5",       "value": clogp, "pass": clogp <= 5},
        {"name": "HBD ≤ 5",         "value": hbd,   "pass": hbd   <= 5},
        {"name": "HBA ≤ 10",        "value": hba,   "pass": hba   <= 10},
    ]
    n_fail = sum(1 for r in rules if not r["pass"])
    summary = "Pass" if n_fail == 0 else ("Borderline" if n_fail == 1 else "Fail")
    return rules, summary


def _pains_alerts(mol) -> list:
    """Return PAINS alert names using RDKit FilterCatalog."""
    try:
        params = FilterCatalogParams()
        params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
        catalog = FilterCatalog(params)
        entries = catalog.GetMatches(mol)
        return [{"name": e.GetDescription(), "description": "PAINS structural alert"} for e in entries]
    except Exception:
        return []


def _sa_score(mol, mw: float) -> tuple:
    """
    Synthetic accessibility heuristic.
    Rough inverse of complexity: penalise rings, stereocenters, MW.
    Returns (score 1-10, class string). Lower = easier to synthesise.
    """
    ring_count = rdMolDescriptors.CalcNumRings(mol)
    spiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)
    stereo = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
    heavy = mol.GetNumHeavyAtoms()

    complexity = (0.02 * mw + 0.5 * ring_count + 1.0 * spiro + 0.7 * stereo)
    score = min(10.0, max(1.0, complexity))
    score = round(score, 1)

    if score <= 3:
        sa_class = "Easy"
    elif score <= 6:
        sa_class = "Moderate"
    else:
        sa_class = "Difficult"

    return score, sa_class


def _fail_fast_score(lipinski_summary: str, num_alerts: int,
                     mw: float, tpsa: float, clogp: float) -> float:
    """
    Composite risk score 0-10 (higher = more concerning).
    """
    score = 0.0
    score += {"Pass": 0, "Borderline": 2, "Fail": 5}.get(lipinski_summary, 0)
    score += min(num_alerts * 2, 4)
    if mw > 600:   score += 2
    if tpsa > 140: score += 1
    if clogp > 5:  score += 1
    if clogp < 0:  score += 1
    return round(min(score, 10.0), 1)


def _decision(fail_fast: float, bbb_prob: float, qed_score: float,
              cns_mpo: float, num_alerts: int) -> tuple:
    """
    Returns (decision: str, rationale: str).
    Decision: 'Progress' | 'Optimize' | 'Kill'
    """
    if fail_fast >= 7 or num_alerts >= 2:
        return (
            "Kill",
            f"High risk score ({fail_fast}/10) or PAINS alerts ({num_alerts}) "
            "indicate major liabilities. Not worth further optimization."
        )
    if qed_score >= 0.6 and cns_mpo >= 4.0 and bbb_prob >= 0.5:
        return (
            "Progress",
            f"Solid drug-likeness (QED {qed_score:.2f}), "
            f"CNS MPO {cns_mpo:.1f}, BBB probability {bbb_prob:.2f}. "
            "Molecule shows promise — continue optimization."
        )
    return (
        "Optimize",
        f"Some properties need improvement: QED {qed_score:.2f}, "
        f"CNS MPO {cns_mpo:.1f}, BBB prob {bbb_prob:.2f}. "
        "Targeted structural changes can address the gaps."
    )


def _optimization_suggestions(clogp: float, mw: float, tpsa: float,
                               hbd: int, bbb_prob: float, cns_mpo: float,
                               qed_score: float, rb: int) -> list:
    """Generate context-specific optimization suggestions."""
    tips = []
    if bbb_prob < 0.5:
        if tpsa > 90:
            tips.append({"text": "Reduce polar surface area (TPSA > 90 Å²) — consider replacing amide/OH with ester or nitrile bioisostere"})
        if hbd > 1:
            tips.append({"text": f"Reduce H-bond donors ({hbd}) — replace NH/OH groups with methylated analogues or bioisosteres"})
        if mw > 400:
            tips.append({"text": f"Reduce molecular weight ({mw:.0f} Da) — remove non-essential substituents"})
    if clogp < 1.0:
        tips.append({"text": "Increase lipophilicity (cLogP too low for good CNS penetration) — add small alkyl or fluoro groups"})
    elif clogp > 4.5:
        tips.append({"text": "Reduce lipophilicity (cLogP too high risks off-target effects) — add polar groups or replace alkyl with heteroatom"})
    if qed_score < 0.6:
        tips.append({"text": f"Drug-likeness (QED {qed_score:.2f}) is suboptimal — simplify structure and reduce MW toward 300-400 Da range"})
    if cns_mpo < 4.0:
        tips.append({"text": f"CNS MPO score ({cns_mpo:.1f}) below threshold — target TPSA < 60, HBD ≤ 1, cLogP 1-3, MW < 400"})
    if rb > 8:
        tips.append({"text": f"Too many rotatable bonds ({rb}) — constrain flexible chains by ring formation or rigidification"})
    if not tips:
        tips.append({"text": "Properties are well-balanced — fine-tune specific targets while preserving overall profile"})
    return tips[:4]  # cap at 4 suggestions


def analyze_local(smiles: str) -> dict:
    """
    Compute a full ADMET profile for a SMILES string using local RDKit.
    Returns a dict in the same shape as the Drug Discovery Triage API response,
    so extract_key_scores() works without any changes.
    """
    if not RDKIT_AVAILABLE:
        return {"error": "RDKit not installed. Run: conda install -c conda-forge rdkit"}

    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        return {"error": f"Invalid SMILES: RDKit could not parse '{smiles[:60]}'"}

    # ── Physicochemical ───────────────────────────────────────────────────────
    canonical = Chem.MolToSmiles(mol)
    mw        = round(Descriptors.ExactMolWt(mol), 2)
    clogp     = round(Crippen.MolLogP(mol), 2)
    tpsa      = round(rdMolDescriptors.CalcTPSA(mol), 1)
    hbd       = rdMolDescriptors.CalcNumHBD(mol)
    hba       = rdMolDescriptors.CalcNumHBA(mol)
    rb        = rdMolDescriptors.CalcNumRotatableBonds(mol)
    fsp3      = round(rdMolDescriptors.CalcFractionCSP3(mol), 3)
    ring_ct   = rdMolDescriptors.CalcNumRings(mol)
    ar_ring   = rdMolDescriptors.CalcNumAromaticRings(mol)

    # ── QED ───────────────────────────────────────────────────────────────────
    qed_val   = round(QED.qed(mol), 3)
    if qed_val >= 0.67:
        qed_class = "Drug-like"
    elif qed_val >= 0.34:
        qed_class = "Moderate"
    else:
        qed_class = "Poor"

    # ── Lipinski ──────────────────────────────────────────────────────────────
    lip_rules, lip_summary = _lipinski_check(mw, clogp, hbd, hba)

    # ── PAINS alerts ──────────────────────────────────────────────────────────
    alerts = _pains_alerts(mol)

    # ── ADMET ─────────────────────────────────────────────────────────────────
    log_s        = round(_esol_log_s(mol, mw, clogp, rb), 2)
    sol_class    = _solubility_class(log_s)
    gi           = _gi_absorption(tpsa, rb, mw, clogp, hbd)
    bbb          = _bbb_probability(clogp, mw, tpsa, hbd)
    cns          = _cns_mpo(clogp, mw, tpsa, hbd)
    cyp          = _cyp_flags(mol)

    # ── Synthetic accessibility & scoring ─────────────────────────────────────
    sa_score, sa_class = _sa_score(mol, mw)
    fail_fast    = _fail_fast_score(lip_summary, len(alerts), mw, tpsa, clogp)
    decision, rationale = _decision(fail_fast, bbb["probability"], qed_val,
                                    cns["score"], len(alerts))
    suggestions  = _optimization_suggestions(clogp, mw, tpsa, hbd,
                                             bbb["probability"], cns["score"],
                                             qed_val, rb)

    return {
        # ── Identity ──────────────────────────────────────────────────────────
        "canonical_smiles": canonical,
        # ── Physicochemical ───────────────────────────────────────────────────
        "molecular_weight": mw,
        "clogp":            clogp,
        "tpsa":             tpsa,
        "hbd":              hbd,
        "hba":              hba,
        "rotatable_bonds":  rb,
        "fraction_sp3":     fsp3,
        "ring_count":       ring_ct,
        "aromatic_ring_count": ar_ring,
        # ── Drug-likeness ─────────────────────────────────────────────────────
        "qed": {"qed_score": qed_val, "qed_class": qed_class},
        "lipinski_rules":   lip_rules,
        "lipinski_summary": lip_summary,
        "alerts":           alerts,
        "synthetic_accessibility_score": sa_score,
        "synthetic_accessibility_class": sa_class,
        "fail_fast_score":  fail_fast,
        "decision":         decision,
        "decision_rationale": rationale,
        "optimization_suggestions": suggestions,
        # ── ADMET ─────────────────────────────────────────────────────────────
        "admet": {
            "solubility": {
                "log_s":            log_s,
                "solubility_class": sol_class,
            },
            "gi_absorption": gi,
            "bbb_penetration": bbb,
            "cns_mpo": cns,
            "cyp_metabolism": cyp,
            "toxicity": {"endpoints": {}},
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Tool 1: Analyze Molecule  (local RDKit — no network required)
# ─────────────────────────────────────────────────────────────────────────────
def call_admet_api(smiles: str, timeout: int = 90) -> dict:
    """
    Analyze a SMILES string and return full ADMET profile.
    Computed locally via RDKit — instant, no API key or network needed.
    (timeout arg kept for API compatibility but unused)
    """
    return analyze_local(smiles.strip())


# ─────────────────────────────────────────────────────────────────────────────
# Tool 2: Validate SMILES  (local RDKit, no API cost)
# ─────────────────────────────────────────────────────────────────────────────
def is_valid_smiles(smiles: str) -> dict:
    """
    Validate a SMILES string using RDKit (fast, offline).
    The agent should always call this before analyze_molecule.
    """
    if not RDKIT_AVAILABLE:
        return {"valid": True, "note": "RDKit unavailable — skipping local validation"}
    try:
        mol = Chem.MolFromSmiles(smiles.strip())
        if mol is None:
            return {"valid": False, "error": "RDKit could not parse this SMILES string"}
        if mol.GetNumHeavyAtoms() == 0:
            return {"valid": False, "error": "Empty SMILES — no atoms found"}
        return {
            "valid": True,
            "canonical_smiles": Chem.MolToSmiles(mol),
            "num_heavy_atoms": mol.GetNumHeavyAtoms(),
        }
    except Exception as e:
        return {"valid": False, "error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# Score Extraction  (flatten full API response → concise dict)
# ─────────────────────────────────────────────────────────────────────────────
def extract_key_scores(response: dict) -> dict:
    """
    Flatten the full API response into the metrics the agent needs to reason about.
    Includes physicochemical properties, ADMET, toxicity, and the API's own decision.
    """
    if "error" in response:
        return response

    admet = response.get("admet", {})
    cyp   = admet.get("cyp_metabolism", {})
    alerts = response.get("alerts", [])
    alert_names = [a.get("name", "") for a in alerts] if isinstance(alerts, list) else []
    suggestions = [s.get("text", "") for s in response.get("optimization_suggestions", [])]

    qed_block = response.get("qed", {})
    qed_score = (qed_block.get("qed_score") or 0) if isinstance(qed_block, dict) else 0

    return {
        # ── Identity ──────────────────────────────────────────────────────────
        "canonical_smiles":   response.get("canonical_smiles", ""),
        # ── Physicochemical ───────────────────────────────────────────────────
        "molecular_weight":   round(response.get("molecular_weight", 0), 1),
        "clogp":              round(response.get("clogp", 0), 2),
        "tpsa":               round(response.get("tpsa", 0), 1),
        "hbd":                response.get("hbd", 0),
        "hba":                response.get("hba", 0),
        "rotatable_bonds":    response.get("rotatable_bonds", 0),
        # ── Drug-likeness ─────────────────────────────────────────────────────
        "qed_score":          round(float(qed_score), 3),
        "qed_class":          qed_block.get("qed_class", "") if isinstance(qed_block, dict) else "",
        "lipinski_summary":   response.get("lipinski_summary", "Unknown"),
        "fail_fast_score":    round(response.get("fail_fast_score", 0), 1),
        "decision":           response.get("decision", "Unknown"),
        "decision_rationale": response.get("decision_rationale", ""),
        # ── ADMET ─────────────────────────────────────────────────────────────
        "solubility_class":   admet.get("solubility", {}).get("solubility_class", "Unknown"),
        "log_s":              admet.get("solubility", {}).get("log_s", None),
        "gi_absorption":      admet.get("gi_absorption", {}).get("absorption", "Unknown"),
        "bbb_penetrates":     admet.get("bbb_penetration", {}).get("penetrates", False),
        "bbb_probability":    round(admet.get("bbb_penetration", {}).get("probability", 0), 3),
        "cns_mpo_score":      round(admet.get("cns_mpo", {}).get("score", 0), 2),
        "cns_class":          admet.get("cns_mpo", {}).get("cns_class", "Unknown"),
        # ── Toxicity ──────────────────────────────────────────────────────────
        "alerts":             alert_names,
        "num_alerts":         len(alert_names),
        "cyp3a4_substrate":   cyp.get("cyp3a4_substrate", False),
        "cyp_inhibitor_risk": cyp.get("cyp_inhibitor_risk", False),
        # ── API's own suggestions (bonus context for the agent) ───────────────
        "api_suggestions":    suggestions,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Tool 3: Compare Candidates
# ─────────────────────────────────────────────────────────────────────────────
def compare_candidates(smiles_list: list, labels: list = None) -> dict:
    """
    Analyze multiple SMILES and return side-by-side key scores.
    Used by the agent to rank candidates across optimization rounds.
    """
    if not labels:
        labels = [f"Candidate {i}" for i in range(len(smiles_list))]

    results = []
    for smiles, label in zip(smiles_list, labels):
        resp = call_admet_api(smiles)
        entry = extract_key_scores(resp) if "error" not in resp else resp
        entry["label"] = label
        results.append(entry)

    return {"candidates": results, "count": len(results)}


# ─────────────────────────────────────────────────────────────────────────────
# Tool Dispatcher  (called by the agent loop)
# ─────────────────────────────────────────────────────────────────────────────
def tool_executor(tool_name: str, tool_input: dict) -> dict:
    """Route a Claude tool_use block to the correct function."""
    if tool_name == "analyze_molecule":
        resp = call_admet_api(tool_input["smiles"])
        return extract_key_scores(resp) if "error" not in resp else resp

    elif tool_name == "validate_smiles":
        return is_valid_smiles(tool_input["smiles"])

    elif tool_name == "compare_candidates":
        return compare_candidates(
            tool_input["smiles_list"],
            tool_input.get("labels"),
        )

    return {"error": f"Unknown tool: '{tool_name}'"}
