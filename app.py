"""
Lead Optimization Agent — Streamlit UI
Run:  streamlit run app.py
"""
import html
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
import streamlit as st
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import anthropic

sys.path.insert(0, os.path.dirname(__file__))
from agent_utils import tool_executor, is_valid_smiles

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Lead Optimization Agent",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Space+Grotesk:wght@400;500;700&display=swap');

:root {
    --bg: #f7f0e8;
    --bg-soft: #fffaf5;
    --panel: rgba(255, 251, 247, 0.84);
    --panel-strong: #fffdf9;
    --ink: #19212a;
    --muted: #5f6b76;
    --line: rgba(25, 33, 42, 0.10);
    --teal: #0f766e;
    --teal-soft: #d7f3ee;
    --sand: #f2e3cf;
    --coral: #d97757;
    --blue: #2563eb;
    --gold: #b7791f;
    --green: #15803d;
    --red: #b42318;
    --shadow: 0 24px 55px rgba(49, 38, 30, 0.10);
}

html, body, [class*="css"] {
    font-family: "Space Grotesk", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    color: var(--ink);
}

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(circle at top left, rgba(215, 243, 238, 0.7), transparent 28%),
        radial-gradient(circle at top right, rgba(242, 227, 207, 0.95), transparent 34%),
        linear-gradient(180deg, #fffaf5 0%, #f8efe6 48%, #f5ebe2 100%);
}

[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 4rem;
    max-width: 1320px;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #193a3b 0%, #224c4d 100%);
    border-right: 1px solid rgba(255, 255, 255, 0.10);
}

[data-testid="stSidebar"] * {
    color: #edf7f5;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: rgba(237, 247, 245, 0.78);
}

[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stTextInput label,
[data-testid="stSidebar"] .stTextArea label,
[data-testid="stSidebar"] .stSlider label {
    color: #f7fbfa;
    font-weight: 600;
}

[data-testid="stSidebar"] .stTextInput input,
[data-testid="stSidebar"] .stTextArea textarea,
[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] > div {
    background: #fffaf7 !important;
    border: 1px solid rgba(255, 255, 255, 0.18);
    border-radius: 16px;
    color: #19212a !important;
    caret-color: #19212a;
}

[data-testid="stSidebar"] .stTextInput input,
[data-testid="stSidebar"] .stTextArea textarea,
[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] * {
    color: #19212a !important;
}

[data-testid="stSidebar"] .stTextInput input::placeholder,
[data-testid="stSidebar"] .stTextArea textarea::placeholder {
    color: #7c8b99 !important;
}

[data-testid="stSidebar"] .stTextInput > div,
[data-testid="stSidebar"] .stTextArea > div,
[data-testid="stSidebar"] .stSelectbox > div {
    background: transparent;
}

[data-testid="stSidebar"] .stButton > button,
[data-testid="stSidebar"] .stDownloadButton > button {
    background: #fffaf7 !important;
    color: #19212a !important;
    border: 1px solid rgba(255, 255, 255, 0.18) !important;
}

[data-testid="stSidebar"] .stButton > button *,
[data-testid="stSidebar"] .stDownloadButton > button * {
    color: #19212a !important;
}

[data-testid="stSidebar"] .stButton > button:hover,
[data-testid="stSidebar"] .stDownloadButton > button:hover {
    background: #fff5ef !important;
    border-color: rgba(255, 255, 255, 0.28) !important;
}

[data-testid="stSidebar"] .stButton > button:disabled,
[data-testid="stSidebar"] .stDownloadButton > button:disabled {
    background: rgba(255, 250, 247, 0.82) !important;
    color: #64748b !important;
    border-color: rgba(255, 255, 255, 0.12) !important;
    opacity: 1 !important;
    box-shadow: none;
}

[data-testid="stSidebar"] .stButton > button:disabled *,
[data-testid="stSidebar"] .stDownloadButton > button:disabled * {
    color: #64748b !important;
}

[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
    background: #fffaf7 !important;
    border: 1px solid rgba(255, 255, 255, 0.18) !important;
}

[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] * {
    color: #19212a !important;
}

[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] small {
    color: #7c8b99 !important;
}

[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button {
    background: #f7f0e8 !important;
    color: #19212a !important;
    border: 1px solid rgba(25, 33, 42, 0.12) !important;
    box-shadow: none;
}

[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button * {
    color: #19212a !important;
}

[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] > div > div {
    background: rgba(255, 255, 255, 0.20);
}

[data-testid="stSidebar"] .stSlider [role="slider"] {
    background: #f7c59f;
    border-color: #f7c59f;
}

.stButton > button {
    border-radius: 999px;
    padding: 0.78rem 1.15rem;
    font-weight: 700;
    letter-spacing: 0.01em;
    border: 1px solid rgba(25, 33, 42, 0.10);
    box-shadow: 0 10px 24px rgba(25, 33, 42, 0.08);
    transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease;
}

.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 14px 28px rgba(25, 33, 42, 0.12);
}

.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #1d847c 0%, #0f766e 100%);
    color: #ffffff;
    border-color: transparent;
}

.stButton > button[kind="secondary"] {
    background: rgba(255, 253, 249, 0.76);
    color: var(--ink);
}

.stTabs [data-baseweb="tab-list"] {
    gap: 0.45rem;
    background: rgba(255, 255, 255, 0.52);
    padding: 0.4rem;
    border: 1px solid var(--line);
    border-radius: 999px;
    width: fit-content;
    box-shadow: 0 10px 24px rgba(25, 33, 42, 0.05);
}

.stTabs [data-baseweb="tab"] {
    border-radius: 999px;
    padding: 0.8rem 1.1rem;
    color: var(--muted);
    font-weight: 600;
    height: auto;
}

.stTabs [aria-selected="true"] {
    background: var(--panel-strong);
    color: var(--ink);
}

.hero-shell {
    position: relative;
    overflow: hidden;
    background:
        radial-gradient(circle at top right, rgba(255, 255, 255, 0.78), transparent 28%),
        linear-gradient(135deg, rgba(215, 243, 238, 0.96) 0%, rgba(255, 250, 245, 0.98) 46%, rgba(242, 227, 207, 0.88) 100%);
    border: 1px solid rgba(25, 33, 42, 0.08);
    border-radius: 32px;
    padding: 2rem;
    box-shadow: var(--shadow);
    margin-bottom: 1.15rem;
}

.hero-shell::after {
    content: "";
    position: absolute;
    width: 240px;
    height: 240px;
    top: -85px;
    right: -60px;
    border-radius: 50%;
    background: rgba(15, 118, 110, 0.08);
}

.hero-grid {
    position: relative;
    z-index: 1;
    display: grid;
    grid-template-columns: minmax(0, 1.4fr) minmax(290px, 0.8fr);
    gap: 1.4rem;
    align-items: start;
}

.hero-eyebrow,
.section-kicker {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.35rem 0.75rem;
    border-radius: 999px;
    background: rgba(255, 255, 255, 0.58);
    border: 1px solid rgba(25, 33, 42, 0.08);
    color: var(--teal);
    font-size: 0.76rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

.hero-title {
    margin: 0.8rem 0 0.7rem;
    font-size: clamp(2.3rem, 4vw, 4.3rem);
    line-height: 0.95;
    letter-spacing: -0.04em;
    color: var(--ink);
}

.hero-title .accent {
    display: block;
    font-family: "Instrument Serif", Georgia, serif;
    font-style: italic;
    font-weight: 400;
    letter-spacing: -0.02em;
}

.hero-text {
    max-width: 48rem;
    font-size: 0.98rem;
    line-height: 1.6;
    color: var(--muted);
    margin: 0;
}

.hero-pill-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.55rem;
    margin-top: 1rem;
}

.hero-pill {
    background: rgba(255, 255, 255, 0.66);
    border: 1px solid rgba(25, 33, 42, 0.08);
    border-radius: 999px;
    padding: 0.5rem 0.8rem;
    font-size: 0.88rem;
    color: var(--ink);
}

.hero-panel,
.overview-card,
.metric-card,
.mol-card,
.best-banner,
.empty-card {
    background: var(--panel);
    backdrop-filter: blur(14px);
    border: 1px solid rgba(25, 33, 42, 0.08);
    box-shadow: var(--shadow);
}

.hero-panel {
    border-radius: 24px;
    padding: 1.35rem;
}

.hero-panel-title {
    font-size: 0.78rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 700;
}

.hero-stat-grid {
    display: grid;
    gap: 0.85rem;
    margin-top: 1rem;
}

.hero-stat {
    display: grid;
    gap: 0.18rem;
    padding-bottom: 0.8rem;
    border-bottom: 1px solid rgba(25, 33, 42, 0.08);
}

.hero-stat:last-child {
    border-bottom: 0;
    padding-bottom: 0;
}

.hero-stat-label {
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--muted);
}

.hero-stat-value {
    font-size: 1.2rem;
    font-weight: 700;
    color: var(--ink);
}

.overview-card,
.empty-card {
    border-radius: 24px;
    padding: 1.3rem 1.35rem;
    min-height: 100%;
}

.overview-label {
    font-size: 0.8rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 700;
}

.overview-title {
    font-size: 1.25rem;
    font-weight: 700;
    color: var(--ink);
    margin: 0.45rem 0 0.65rem;
}

.overview-text,
.small-note,
.check-copy {
    color: var(--muted);
    line-height: 1.55;
    font-size: 0.9rem;
}

.mono-preview {
    margin-top: 0.9rem;
    padding: 0.8rem 0.95rem;
    border-radius: 18px;
    background: rgba(25, 33, 42, 0.04);
    border: 1px solid rgba(25, 33, 42, 0.08);
    font-family: "SFMono-Regular", ui-monospace, monospace;
    font-size: 0.82rem;
    overflow-wrap: anywhere;
}

.brief-lead {
    color: var(--muted);
    line-height: 1.6;
    font-size: 0.95rem;
    margin: 0.15rem 0 0;
}

.brief-targets {
    margin-top: 0.9rem;
    display: grid;
    gap: 0.5rem;
}

.brief-target {
    display: flex;
    gap: 0.6rem;
    align-items: flex-start;
    color: var(--ink);
    font-size: 0.93rem;
    line-height: 1.45;
}

.brief-target::before {
    content: "";
    width: 8px;
    height: 8px;
    border-radius: 999px;
    margin-top: 0.38rem;
    flex-shrink: 0;
    background: linear-gradient(135deg, #0f766e 0%, #d97757 100%);
}

.brief-meta {
    margin-top: 0.9rem;
    font-size: 0.82rem;
    color: #7c8b99;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-weight: 700;
}

.readiness-list {
    display: grid;
    gap: 0.9rem;
    margin-top: 1rem;
}

.readiness-item {
    display: grid;
    grid-template-columns: auto 1fr;
    gap: 0.75rem;
    align-items: start;
}

.readiness-chip {
    min-width: 78px;
    text-align: center;
    border-radius: 999px;
    padding: 0.42rem 0.65rem;
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

.chip-ready { background: rgba(21, 128, 61, 0.12); color: var(--green); }
.chip-warn  { background: rgba(217, 119, 87, 0.14); color: var(--coral); }

.section-heading {
    margin: 1.8rem 0 1rem;
}

.section-title {
    margin: 0.5rem 0 0;
    font-size: 1.55rem;
    line-height: 1.1;
    letter-spacing: -0.03em;
    color: var(--ink);
}

.section-copy {
    color: var(--muted);
    margin: 0.35rem 0 0;
    max-width: 42rem;
    font-size: 0.92rem;
}

.metric-card {
    border-radius: 24px;
    padding: 1.35rem 1.2rem;
    text-align: center;
}

.metric-label {
    font-size: 0.76rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.45rem;
    font-weight: 700;
}

.metric-before {
    font-size: 0.88rem;
    color: var(--muted);
}

.metric-after {
    font-size: 2rem;
    line-height: 1;
    font-weight: 700;
    margin: 0.4rem 0 0.3rem;
    letter-spacing: -0.04em;
}

.metric-delta {
    font-size: 0.9rem;
    font-weight: 700;
}

.best-banner {
    border-radius: 26px;
    padding: 1.25rem 1.35rem;
    margin-top: 1rem;
}

.best-banner-label {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.35rem 0.75rem;
    border-radius: 999px;
    background: rgba(21, 128, 61, 0.12);
    color: var(--green);
    font-size: 0.76rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 700;
}

.best-banner-title {
    font-size: 1.4rem;
    line-height: 1.1;
    margin: 0.75rem 0 0.45rem;
    color: var(--ink);
}

.best-banner-copy {
    color: var(--muted);
    margin: 0;
    font-size: 0.93rem;
    line-height: 1.55;
}

.best-banner-code {
    margin-top: 0.9rem;
    padding: 0.9rem 1rem;
    border-radius: 18px;
    background: rgba(25, 33, 42, 0.05);
    border: 1px solid rgba(25, 33, 42, 0.08);
    font-family: "SFMono-Regular", ui-monospace, monospace;
    overflow-wrap: anywhere;
}

.result-pill-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.55rem;
    margin-top: 0.9rem;
}

.result-pill {
    border-radius: 999px;
    padding: 0.45rem 0.75rem;
    background: rgba(255, 255, 255, 0.7);
    border: 1px solid rgba(25, 33, 42, 0.08);
    color: var(--ink);
    font-size: 0.87rem;
}

.mol-card {
    border-radius: 28px;
    padding: 1.45rem;
    margin-bottom: 1.25rem;
    border-left: 5px solid rgba(25, 33, 42, 0.14);
}

.mol-card.best  { border-left-color: var(--green); }
.mol-card.good  { border-left-color: var(--blue); }
.mol-card.start { border-left-color: #7c8b99; }

.mol-structure-panel {
    display: flex;
    align-items: center;
    justify-content: center;
    min-height: 320px;
    padding: 1rem;
    border-radius: 24px;
    background: linear-gradient(180deg, #fbfcfe 0%, #eef3f8 100%);
    border: 1px solid rgba(71, 85, 105, 0.16);
    overflow: hidden;
}

.mol-structure-panel svg {
    width: 100%;
    height: auto;
    max-height: 300px;
}

.mol-structure-fallback {
    width: 100%;
    padding: 1rem;
    border-radius: 18px;
    background: rgba(255, 255, 255, 0.7);
    border: 1px dashed rgba(25, 33, 42, 0.14);
}

.change-note {
    margin-top: 0.55rem;
    font-size: 0.78rem;
    color: #b45309;
    font-weight: 700;
    letter-spacing: 0.03em;
}

.round-badge {
    display: inline-block;
    border-radius: 999px;
    padding: 0.42rem 0.78rem;
    font-size: 0.76rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 0.35rem;
}

.badge-best    { background: rgba(21, 128, 61, 0.12); color: var(--green); }
.badge-start   { background: rgba(25, 33, 42, 0.07); color: #596674; }
.badge-attempt { background: rgba(37, 99, 235, 0.10); color: #1d4ed8; }

.score-section { margin: 0.75rem 0 0.1rem; }

.score-row {
    display: flex;
    align-items: center;
    margin: 0.72rem 0;
    gap: 0.7rem;
}

.score-label {
    font-size: 0.86rem;
    color: var(--muted);
    width: 170px;
    flex-shrink: 0;
}

.score-bar-bg {
    flex: 1;
    background: rgba(25, 33, 42, 0.08);
    border-radius: 999px;
    height: 10px;
    overflow: hidden;
}

.score-bar {
    height: 100%;
    border-radius: 999px;
    transition: width 0.3s ease;
}

.score-value {
    font-size: 0.88rem;
    font-weight: 700;
    width: 58px;
    text-align: right;
    flex-shrink: 0;
}

.score-delta {
    font-size: 0.76rem;
    width: 52px;
    text-align: right;
    flex-shrink: 0;
}

.green { color: var(--green); }
.red   { color: var(--red); }
.grey  { color: #8c99a5; }

.alert-pill {
    display: inline-block;
    border-radius: 999px;
    padding: 0.42rem 0.72rem;
    font-size: 0.76rem;
    font-weight: 700;
    margin-top: 0.35rem;
}

.pill-green  { background: rgba(21, 128, 61, 0.12); color: var(--green); }
.pill-yellow { background: rgba(183, 121, 31, 0.14); color: var(--gold); }
.pill-red    { background: rgba(180, 35, 24, 0.12); color: var(--red); }

.reasoning-box {
    background: linear-gradient(180deg, rgba(255, 255, 255, 0.68), rgba(255, 255, 255, 0.82));
    border-radius: 22px;
    padding: 0.9rem 1rem;
    margin-top: 1rem;
    border: 1px solid rgba(25, 33, 42, 0.08);
}

.reasoning-title {
    font-size: 0.76rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--teal);
    margin-bottom: 0.45rem;
}

.reasoning-text {
    font-size: 0.92rem;
    color: #32404d;
    line-height: 1.65;
}

.page-subtitle {
    font-size: 0.98rem;
    color: var(--muted);
    margin-top: -0.2rem;
    margin-bottom: 0;
}

.empty-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 1rem;
    margin-top: 0.8rem;
}

.empty-step {
    font-size: 0.76rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--teal);
}

.empty-title {
    font-size: 1.1rem;
    font-weight: 700;
    margin: 0.5rem 0 0.45rem;
    color: var(--ink);
}

.empty-copy {
    margin: 0;
    color: var(--muted);
    line-height: 1.55;
    font-size: 0.9rem;
}

div[data-testid="stAlert"] {
    border-radius: 20px;
    border: 1px solid rgba(25, 33, 42, 0.08);
}

@media (max-width: 980px) {
    .hero-grid,
    .empty-grid {
        grid-template-columns: 1fr;
    }

    .score-label {
        width: 120px;
    }
}
</style>
""", unsafe_allow_html=True)


# ─── Helpers ──────────────────────────────────────────────────────────────────
def score_color(val, low, mid, high, invert=False):
    """Return hex color based on thresholds."""
    if invert:
        val = -val; low, mid, high = -high, -mid, -low
    if val >= high:   return "#16a34a"
    if val >= mid:    return "#f59e0b"
    return "#dc2626"

def bar_html(label, value, display, bar_pct, color, delta_str=""):
    return f"""
<div class="score-row">
  <span class="score-label">{label}</span>
  <div class="score-bar-bg">
    <div class="score-bar" style="width:{bar_pct:.0f}%;background:{color}"></div>
  </div>
  <span class="score-value" style="color:{color}">{display}</span>
  <span class="score-delta {'green' if delta_str.startswith('+') else 'red' if delta_str.startswith('-') else 'grey'}">{delta_str}</span>
</div>"""

def mol_scores_html(c, start=None):
    """Render score bars for one candidate."""
    def delta(key, fmt="+.2f"):
        if start is None: return ""
        d = (c.get(key) or 0) - (start.get(key) or 0)
        return f"{d:{fmt}}" if d != 0 else ""

    bbb   = (c.get("bbb_probability") or 0) * 100
    cns   = c.get("cns_mpo_score") or 0
    qed   = (c.get("qed_score") or 0) * 100
    flex  = c.get("rotatable_bonds") or 0
    alerts= c.get("num_alerts") or 0

    out = '<div class="score-section">'
    out += bar_html("Brain Penetration",  bbb,  f"{bbb:.0f}%",  bbb,
                    score_color(bbb, 50, 70, 80), delta("bbb_probability", "+.0%").replace("%","pp") if start else "")
    out += bar_html("CNS Activity Score", cns,  f"{cns:.1f}/5", cns/5*100,
                    score_color(cns, 3, 4, 4.5), delta("cns_mpo_score") if start else "")
    out += bar_html("Drug Likeness",      qed,  f"{qed:.0f}%",  qed,
                    score_color(qed, 49, 60, 67), delta("qed_score", "+.0%").replace("%","pp") if start else "")
    out += bar_html("Flexibility",        flex, f"{flex} bonds", max(0, 100-flex*10),
                    score_color(flex, 9, 7, 5, invert=True), delta("rotatable_bonds", "+d") if start else "")

    # Safety pill
    alert_color = "pill-green" if alerts == 0 else ("pill-yellow" if alerts <= 2 else "pill-red")
    alert_text  = "No safety alerts" if alerts == 0 else f"{alerts} safety alert{'s' if alerts>1 else ''}"
    out += f'<div style="margin-top:10px"><span class="alert-pill {alert_color}">{alert_text}</span></div>'
    out += "</div>"
    return out

def escape_html(text):
    return html.escape(str(text or "")).replace("\n", "<br>")

def truncate_text(text, limit=220):
    clean = " ".join(str(text or "").split())
    if len(clean) <= limit:
        return clean
    return clean[: limit - 1].rstrip() + "…"

def render_section_heading(title, subtitle="", kicker="Overview"):
    st.markdown(
        f"""
<div class="section-heading">
  <span class="section-kicker">{escape_html(kicker)}</span>
  <h2 class="section-title">{escape_html(title)}</h2>
  {f'<p class="section-copy">{escape_html(subtitle)}</p>' if subtitle else ''}
</div>
""",
        unsafe_allow_html=True,
    )

def readiness_item(label, detail, ready):
    chip_class = "chip-ready" if ready else "chip-warn"
    chip_text = "Ready" if ready else "Needed"
    return f"""
<div class="readiness-item">
  <div class="readiness-chip {chip_class}">{chip_text}</div>
  <div>
    <div class="overview-label">{escape_html(label)}</div>
    <div class="check-copy">{escape_html(detail)}</div>
  </div>
</div>
"""

def summarize_reasoning(text, is_start=False, limit=190):
    clean = str(text or "")
    clean = re.sub(r"```.*?```", " ", clean, flags=re.S)
    clean = re.sub(r"^#{1,6}\s*", "", clean, flags=re.M)
    clean = clean.replace("---", " ")
    clean = clean.replace("**", "").replace("`", "")
    clean = re.sub(r"\s+", " ", clean).strip()

    if not clean:
        return ""

    if "Rationale:" in clean:
        rationale = clean.split("Rationale:", 1)[1].strip()
    else:
        rationale = clean

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", rationale) if s.strip()]
    if not sentences:
        return truncate_text(clean, limit)

    if is_start:
        return truncate_text(sentences[0], limit)

    keywords = (
        "replace",
        "switch",
        "move",
        "moving",
        "add",
        "adding",
        "remove",
        "reduce",
        "shift",
        "place",
        "placing",
        "orientation",
        "scaffold",
    )
    selected = next((s for s in sentences if any(k in s.lower() for k in keywords)), sentences[0])
    selected = re.sub(r"^(let's|let me)\s+", "", selected, flags=re.I)
    selected = re.sub(r"^very close!?[\s,:-]*", "", selected, flags=re.I)
    selected = re.sub(r"^rationale:\s*", "", selected, flags=re.I)
    return truncate_text(selected, limit)

def _metric_priority_phrase(current, previous):
    if previous is None:
        return ""

    improvements = []
    bbb_delta = (current.get("bbb_probability") or 0) - (previous.get("bbb_probability") or 0)
    cns_delta = (current.get("cns_mpo_score") or 0) - (previous.get("cns_mpo_score") or 0)
    qed_delta = (current.get("qed_score") or 0) - (previous.get("qed_score") or 0)
    flex_delta = (previous.get("rotatable_bonds") or 0) - (current.get("rotatable_bonds") or 0)
    alert_delta = (previous.get("num_alerts") or 0) - (current.get("num_alerts") or 0)

    if bbb_delta > 0.015:
        improvements.append("improve brain penetration")
    if cns_delta > 0.05:
        improvements.append("lift CNS MPO")
    if qed_delta > 0.015:
        improvements.append("improve overall drug-likeness")
    if flex_delta > 0:
        improvements.append("reduce conformational flexibility")
    if alert_delta > 0:
        improvements.append("reduce structural risk")

    if not improvements:
        return "fine-tune the CNS profile"
    if len(improvements) == 1:
        return improvements[0]
    return f"{improvements[0]} and {improvements[1]}"

def _structural_change_phrase(reasoning):
    text = str(reasoning or "").lower()

    checks = [
        (("amide", "nitrile"), "replaced the amide with a nitrile handle"),
        (("n-methyl",), "N-methylated the side-chain amine"),
        (("fluoro",), "introduced a fluorine on the side chain"),
        (("fluorine",), "introduced a fluorine on the side chain"),
        (("methyl",), "added a small methyl substituent"),
        (("pyridine",), "swapped the phenyl ring for a pyridine"),
        (("rigid",), "rigidified the scaffold"),
        (("fused ring",), "rigidified the aryl region with a fused ring"),
        (("naphthal",), "expanded the aryl system to a fused bicyclic ring"),
        (("bioisostere",), "made a bioisosteric swap"),
    ]
    for needles, phrase in checks:
        if all(needle in text for needle in needles):
            return phrase

    match = re.search(r"replace(?:d)?\s+(.+?)\s+with\s+(.+?)(?:[.,;]|$)", text)
    if match:
        before = truncate_text(match.group(1), 35)
        after = truncate_text(match.group(2), 35)
        return f"replaced {before} with {after}"

    return "changed the highlighted region"

def summarize_candidate_change(current, previous=None):
    reasoning = current.get("reasoning") or ""
    if previous is None:
        qed  = (current.get("qed_score") or 0) * 100
        cns  = current.get("cns_mpo_score") or 0
        bbb  = (current.get("bbb_probability") or 0) * 100
        logs = current.get("log_s") or 0
        flex = current.get("rotatable_bonds") or 0
        return truncate_text(
            f"Baseline readout: drug-likeness {qed:.0f}%, CNS score {cns:.2f}/5, "
            f"brain penetration {bbb:.0f}%, solubility logS {logs:.2f}, "
            f"{flex} rotatable bonds. This is the starting point for optimization.",
            220,
        )

    change_phrase = _structural_change_phrase(reasoning)
    why_phrase = _metric_priority_phrase(current, previous)
    return truncate_text(
        f"We {change_phrase} to {why_phrase}.",
        210,
    )

def brief_title_from_name(name):
    text = str(name or "").strip()
    if not text or text == "Custom molecule":
        return "Custom lead brief"
    return text.replace("→", "to")

def brief_body_html(goal_text):
    text = str(goal_text or "").strip()
    if not text:
        return '<p class="brief-lead">Add the target profile you want the agent to optimise toward.</p>'

    sections = [part.strip() for part in text.split("\n\n") if part.strip()]
    intro = sections[0] if sections else text
    intro = intro.replace("Targets:", "").strip()
    intro_lines = [line.strip(" -") for line in intro.splitlines() if line.strip()]
    intro_text = " ".join(intro_lines[:2]).strip()

    targets = []
    for section in sections[1:]:
        for line in section.splitlines():
            stripped = line.strip()
            if stripped.startswith("-"):
                targets.append(stripped.lstrip("- ").strip())

    if not targets:
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("-"):
                targets.append(stripped.lstrip("- ").strip())

    html_parts = [f'<p class="brief-lead">{escape_html(truncate_text(intro_text, 260))}</p>']
    if targets:
        targets_html = "".join(
            f'<div class="brief-target">{escape_html(truncate_text(item, 110))}</div>'
            for item in targets[:4]
        )
        html_parts.append(f'<div class="brief-targets">{targets_html}</div>')
    return "".join(html_parts)

def normalize_candidates(raw_candidates):
    if not isinstance(raw_candidates, list):
        raise ValueError("Saved run must contain a list of candidates.")

    normalized = []
    for idx, item in enumerate(raw_candidates):
        if not isinstance(item, dict):
            continue
        row = dict(item)
        row["mol_index"] = row.get("mol_index", row.get("round", idx))
        row["input_smiles"] = row.get("input_smiles") or row.get("canonical_smiles") or ""
        row["reasoning"] = row.get("reasoning") or ""
        normalized.append(row)

    if not normalized:
        raise ValueError("No valid candidates found in saved run.")
    return normalized

def build_run_payload(smiles, goal, candidates, preset_name, source="live"):
    return {
        "version": 1,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "source": source,
        "preset_name": preset_name,
        "starting_smiles": smiles,
        "goal": goal,
        "candidate_count": len(candidates),
        "candidates": candidates,
    }

def load_run_payload(payload):
    if isinstance(payload, list):
        candidates = normalize_candidates(payload)
        meta = {
            "preset_name": "Saved run",
            "starting_smiles": candidates[0].get("input_smiles", ""),
            "goal": "",
            "saved_at": "",
            "source": "upload",
        }
        return candidates, meta

    if not isinstance(payload, dict):
        raise ValueError("Unsupported saved run format.")

    candidates = normalize_candidates(payload.get("candidates") or payload.get("results") or [])
    meta = {
        "preset_name": payload.get("preset_name") or "Saved run",
        "starting_smiles": payload.get("starting_smiles") or payload.get("smiles") or candidates[0].get("input_smiles", ""),
        "goal": payload.get("goal") or "",
        "saved_at": payload.get("saved_at") or "",
        "source": payload.get("source") or "upload",
    }
    return candidates, meta

def apply_loaded_run(payload, source_label):
    candidates, meta = load_run_payload(payload)
    meta["source_label"] = source_label
    st.session_state.candidates = candidates
    st.session_state.completed = True
    st.session_state.run_meta = meta
    st.session_state.run_notice = f"Loaded {len(candidates)} candidates from {source_label}."

def save_run_snapshot(payload, runs_dir, latest_path):
    runs_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_path = runs_dir / f"run_{stamp}.json"
    text = json.dumps(payload, indent=2)
    latest_path.write_text(text, encoding="utf-8")
    run_path.write_text(text, encoding="utf-8")
    return run_path

def mol_svg_markup(smiles, previous_smiles=None):
    """Return inline SVG markup for a SMILES string, optionally highlighting the changed region."""
    try:
        from rdkit import Chem
        from rdkit.Chem import rdFMCS
        from rdkit.Chem.Draw import rdMolDraw2D
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, False

        highlight_atoms = []
        highlight_bonds = []
        highlight_atom_colors = {}
        highlight_bond_colors = {}
        highlight_atom_radii = {}

        if previous_smiles:
            previous = Chem.MolFromSmiles(previous_smiles)
            if previous is not None:
                mcs = rdFMCS.FindMCS(
                    [previous, mol],
                    ringMatchesRingOnly=True,
                    completeRingsOnly=True,
                    atomCompare=rdFMCS.AtomCompare.CompareElements,
                    bondCompare=rdFMCS.BondCompare.CompareOrder,
                    timeout=2,
                )
                if mcs and mcs.smartsString:
                    patt = Chem.MolFromSmarts(mcs.smartsString)
                    if patt is not None:
                        match = mol.GetSubstructMatch(patt)
                        if match:
                            core_atoms = set(match)
                            highlight_atoms = [a.GetIdx() for a in mol.GetAtoms() if a.GetIdx() not in core_atoms]
                            highlight_bonds = [
                                b.GetIdx()
                                for b in mol.GetBonds()
                                if b.GetBeginAtomIdx() in highlight_atoms or b.GetEndAtomIdx() in highlight_atoms
                            ]
                            highlight_atom_colors = {idx: (0.98, 0.83, 0.18) for idx in highlight_atoms}
                            highlight_bond_colors = {idx: (0.98, 0.83, 0.18) for idx in highlight_bonds}
                            highlight_atom_radii = {idx: 0.5 for idx in highlight_atoms}

        Chem.rdDepictor.Compute2DCoords(mol)
        drawer = rdMolDraw2D.MolDraw2DSVG(420, 300)
        opts = drawer.drawOptions()
        opts.bondLineWidth = 2
        opts.padding = 0.05
        opts.multipleBondOffset = 0.18
        opts.fixedBondLength = 42
        opts.clearBackground = False
        opts.fillHighlights = False
        opts.highlightRadius = 0.3
        opts.highlightBondWidthMultiplier = 18
        opts.atomHighlightsAreCircles = True
        opts.useDefaultAtomPalette()
        opts.updateAtomPalette({
            6: (0.11, 0.16, 0.23),  # carbon
            7: (0.10, 0.35, 0.82),  # nitrogen
            8: (0.84, 0.15, 0.16),  # oxygen
            9: (0.00, 0.62, 0.69),  # fluorine
            15: (0.58, 0.20, 0.78), # phosphorus
            16: (0.84, 0.45, 0.08), # sulfur
            17: (0.09, 0.55, 0.35), # chlorine
            35: (0.65, 0.25, 0.16), # bromine
            53: (0.43, 0.34, 0.78), # iodine
        })
        opts.setHighlightColour((0.95, 0.69, 0.12))
        opts.setBackgroundColour((1.0, 1.0, 1.0, 0.0))
        rdMolDraw2D.PrepareAndDrawMolecule(
            drawer,
            mol,
            highlightAtoms=highlight_atoms,
            highlightAtomColors=highlight_atom_colors,
            highlightAtomRadii=highlight_atom_radii,
            highlightBonds=highlight_bonds,
            highlightBondColors=highlight_bond_colors,
        )
        drawer.FinishDrawing()
        svg = drawer.GetDrawingText()
        svg = re.sub(r"<\?xml.*?\?>", "", svg).strip()
        return svg, bool(highlight_atoms)
    except Exception:
        return None, False


# ─── Constants ────────────────────────────────────────────────────────────────
PRESETS = {
    "Atenolol → Brain Penetration": {
        "smiles": "CC(C)NCC(O)COc1ccc(CC(N)=O)cc1",
        "goal": (
            "Transform this Atenolol scaffold toward a brain-penetrating profile.\n"
            "Reference: Propranolol (brain penetration 90%, CNS score 5.45, drug likeness 84%).\n\n"
            "Targets:\n"
            "  - Brain penetration > 80%  (currently ~18%)\n"
            "  - CNS Activity Score > 4.0  (currently ~2.3)\n"
            "  - Drug likeness > 75%       (currently ~64%)\n"
            "  - Flexibility ≤ 6 bonds     (currently 8)\n"
            "  - Zero safety alerts\n"
            "  - Molecular weight < 400 Da\n\n"
            "The para-amide group (-CC(N)=O) is the main polarity issue. "
            "Preserve the core beta-blocker scaffold (aryloxypropanolamine)."
        ),
    },
    "Aspirin → CNS Drug Profile": {
        "smiles": "CC(=O)Oc1ccccc1C(=O)O",
        "goal": (
            "Aspirin contains a carboxylic acid (-COOH) that limits CNS penetration.\n"
            "Improve the CNS drug profile while retaining the salicylate scaffold.\n\n"
            "Targets:\n"
            "  - CNS Activity Score > 4.0  (currently ~3.5)\n"
            "  - Brain penetration > 70%   (currently ~56%)\n"
            "  - Drug likeness > 70%        (currently ~55%)\n"
            "  - Keep molecular weight < 300 Da\n\n"
            "The free carboxylic acid is the main liability — consider replacing with "
            "a bioisostere (amide, ester, tetrazole) or masking it. "
            "The acetyl ester can also be explored."
        ),
    },
    "Ibuprofen → Aqueous Solubility": {
        "smiles": "CC(C)Cc1ccc(C(C)C(=O)O)cc1",
        "goal": (
            "Ibuprofen is a BCS Class II drug — good permeability but poor aqueous solubility.\n"
            "Improve solubility while preserving anti-inflammatory drug-likeness.\n\n"
            "Targets:\n"
            "  - Aqueous solubility logS > -2.0  (currently ~-3.1, poorly soluble)\n"
            "  - Drug likeness > 80%              (currently ~82%, maintain)\n"
            "  - Keep GI absorption: High\n"
            "  - Molecular weight < 300 Da\n\n"
            "High cLogP (~3.1) and the isobutyl chain drive poor solubility. "
            "Consider adding a hydroxyl or reducing the alkyl chain. "
            "Preserve the propionic acid pharmacophore for COX inhibition."
        ),
    },
    "Custom molecule": {"smiles": "", "goal": ""},
}

RUNS_DIR = Path(__file__).resolve().parent / "saved_runs"
LATEST_RUN_PATH = RUNS_DIR / "latest_run.json"

SYSTEM_PROMPT = """
You are a senior medicinal chemist AI. Your job is to iteratively improve a drug
candidate toward the stated optimization goal, while keeping it drug-like and safe.

## Tools
1. validate_smiles(smiles)          — ALWAYS call before analyze_molecule
2. analyze_molecule(smiles)         — Get full ADMET profile (local, instant)
3. compare_candidates(smiles_list)  — Compare multiple candidates side by side

## Strategy per round
  1. Read the goal carefully — identify which property is furthest from target
  2. Choose ONE structural change that addresses it (clear SAR reasoning)
  3. Validate the new SMILES before analyzing it
  4. Compare to previous rounds every 2 iterations
  5. Stop when the goal is fully met or no improvement across 2 consecutive rounds

## Key structure–property rules
  - Brain penetration / CNS: lower H-bond donors, TPSA < 90 Å², cLogP 1–4
  - Aqueous solubility: lower cLogP, add polar groups (OH, NH), shorten alkyl chains
  - Oral absorption: TPSA ≤ 140, rotatable bonds ≤ 10 (Veber rules)
  - Drug likeness (QED): MW 200–450, LogP 0–5, fewer rings and rotatable bonds
  - Bioisosteres: -CONH₂→-CN, -COOH→-tetrazole, -OH→-F, phenyl→pyridine
  - Remove alerts: avoid quinones, catechols, Michael acceptors, PAINS motifs

## Style
  - Explain your chemical reasoning in plain, clear language
  - Describe WHAT you changed, WHY, and WHAT improvement you expect
  - End with the best candidate SMILES and a plain-language summary
""".strip()

TOOLS = [
    {
        "name": "validate_smiles",
        "description": "Validate a SMILES string (local, instant). Always call before analyze_molecule.",
        "input_schema": {"type":"object","properties":{"smiles":{"type":"string"}},"required":["smiles"]},
    },
    {
        "name": "analyze_molecule",
        "description": "Get full ADMET profile (local RDKit, instant).",
        "input_schema": {"type":"object","properties":{"smiles":{"type":"string"}},"required":["smiles"]},
    },
    {
        "name": "compare_candidates",
        "description": "Compare multiple molecules side by side.",
        "input_schema": {
            "type":"object",
            "properties":{
                "smiles_list":{"type":"array","items":{"type":"string"}},
                "labels":{"type":"array","items":{"type":"string"}},
            },
            "required":["smiles_list"],
        },
    },
]

# ─── Session state ─────────────────────────────────────────────────────────────
for k, v in {"candidates": [], "completed": False, "run_meta": {}, "run_notice": "", "render_seq": 0}.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─── Agent runner ──────────────────────────────────────────────────────────────
def run_agent(smiles, goal, max_tool_calls, api_key, status_placeholder,
              results_placeholder=None, journey_placeholder=None):
    client = anthropic.Anthropic(api_key=api_key)
    messages = [{"role":"user","content":(
        f"Starting molecule SMILES: {smiles}\n\nOptimization goal:\n{goal}\n\n"
        "Analyze the starting molecule first, then propose and evaluate structural "
        "modifications round by round. Explain your chemical reasoning clearly for "
        "each change. End with the best candidate and a plain-language summary."
    )}]
    candidates   = []
    tool_count   = 0
    pending_text = ""   # agent text accumulated before next analyze_molecule call
    mol_index    = 0    # counts analyze_molecule calls (0 = starting molecule)

    while True:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )
        messages.append({"role":"assistant","content":response.content})

        # Accumulate agent reasoning text
        for block in response.content:
            if hasattr(block, "text") and block.text.strip():
                pending_text += block.text.strip() + "\n\n"

        if response.stop_reason == "end_turn":
            status_placeholder.success("✅ Optimization complete!")
            break

        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue

                tool_count += 1
                label = "Starting molecule" if mol_index == 0 else f"Attempt {mol_index}"
                status_placeholder.info(f"🔬 Evaluating {label}…  (step {tool_count})")

                result = tool_executor(block.name, block.input)

                # Capture each analyzed molecule with its reasoning
                if block.name == "analyze_molecule" and "error" not in result:
                    candidates.append({
                        "mol_index":   mol_index,
                        "input_smiles": block.input["smiles"],
                        "reasoning":   pending_text.strip(),
                        **result,
                    })
                    mol_index    += 1
                    pending_text  = ""
                    st.session_state.candidates = list(candidates)
                    if results_placeholder is not None and journey_placeholder is not None:
                        render_tab_views(results_placeholder, journey_placeholder, candidates)

                tool_results.append({
                    "type":"tool_result",
                    "tool_use_id":block.id,
                    "content":json.dumps(result, default=str),
                })
            messages.append({"role":"user","content":tool_results})

        if tool_count >= max_tool_calls:
            status_placeholder.warning(f"Reached maximum steps ({max_tool_calls}).")
            break

    st.session_state.candidates = candidates
    st.session_state.completed  = True


def render_results_tab(candidates, render_token=0):
    if not candidates:
        st.info("Run the optimisation to see results here.")
        return

    df = pd.DataFrame(candidates)
    start = df.iloc[0].to_dict()
    best = df.loc[df["cns_mpo_score"].idxmax()].to_dict() if "cns_mpo_score" in df.columns else df.iloc[-1].to_dict()

    render_section_heading(
        "What improved from the starting scaffold",
        "The key deltas first.",
        "Results",
    )

    def metric_card(label, before, after, fmt, unit="", invert=False):
        delta = after - before
        better = (delta < 0) if invert else (delta > 0)
        arrow  = "▲" if better else ("▼" if delta != 0 else "—")
        color  = "#15803d" if better else ("#b42318" if delta != 0 else "#8c99a5")
        sign   = "+" if delta > 0 else ""
        return f"""
<div class="metric-card">
  <div class="metric-label">{label}</div>
  <div class="metric-before">Before: {before:{fmt}}{unit}</div>
  <div class="metric-after" style="color:{color}">{after:{fmt}}{unit}</div>
  <div class="metric-delta" style="color:{color}">{arrow} {sign}{delta:{fmt}}{unit}</div>
</div>"""

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(metric_card("Brain Penetration", start.get("bbb_probability", 0) * 100, best.get("bbb_probability", 0) * 100, ".0f", "%"), unsafe_allow_html=True)
    c2.markdown(metric_card("CNS Activity Score", start.get("cns_mpo_score", 0), best.get("cns_mpo_score", 0), ".2f", "/5"), unsafe_allow_html=True)
    c3.markdown(metric_card("Drug Likeness", start.get("qed_score", 0) * 100, best.get("qed_score", 0) * 100, ".0f", "%"), unsafe_allow_html=True)
    c4.markdown(metric_card("Molecular Flexibility", start.get("rotatable_bonds", 0), best.get("rotatable_bonds", 0), "d", " bonds", invert=True), unsafe_allow_html=True)

    st.write("")

    render_section_heading(
        "Trajectory across optimisation rounds",
        "See whether performance compounded or just shifted.",
        "Progress",
    )

    rounds = list(range(len(df)))
    labels = ["Start"] + [f"Attempt {i}" for i in range(1, len(df))]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.patch.set_facecolor("#f7f0e8")

    chart_metrics = [
        ("bbb_probability", "Brain Penetration (%)",  "#2563eb", 0.80, lambda v: v * 100),
        ("cns_mpo_score",   "CNS Activity Score",     "#0f766e", 4.0,  lambda v: v),
        ("qed_score",       "Drug Likeness (%)",      "#d97757", 0.75, lambda v: v * 100),
    ]
    for ax, (col, title, color, target, transform) in zip(axes, chart_metrics):
        ax.set_facecolor("#fffdf9")
        vals = [transform(v) for v in df[col].tolist()]
        tgt  = transform(target)
        ax.plot(rounds, vals, "o-", color=color, linewidth=2.5, markersize=9, zorder=3)
        ax.axhline(y=tgt, color="#b7791f", linestyle="--", alpha=0.8, label=f"Target ({transform(target):.0f}{'%' if '%' in title else ''})")
        ax.fill_between(rounds, vals, alpha=0.08, color=color)
        for r, v in zip(rounds, vals):
            ax.annotate(f"{v:.1f}", (r, v), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=8.5, color=color, fontweight="bold")
        ax.set_title(title, fontweight="bold", fontsize=11, color="#19212a", pad=10)
        ax.set_xticks(rounds)
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
        ax.tick_params(colors="#5f6b76")
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_edgecolor("#d8cdc0")
        ax.legend(fontsize=8, framealpha=0.8)
        ax.grid(True, alpha=0.18, color="#c7b8a8")

    plt.tight_layout(pad=2)
    st.pyplot(fig, use_container_width=True)
    plt.close()

    render_section_heading(
        "Overall profile, start versus best",
        "A fast balance check.",
        "Profile fit",
    )

    radar_props = [
        ("qed_score",       "Drug Likeness",      0, 1,   True),
        ("bbb_probability", "Brain Penetration",  0, 1,   True),
        ("cns_mpo_score",   "CNS Activity",       0, 5,   True),
        ("gi_absorption",   "Oral Absorption",    0, 1,   True),
        ("log_s",           "Solubility",        -10, 2,  True),
        ("num_alerts",      "Safety",             0, 5,   False),
    ]

    def norm(row, col, lo, hi, pos):
        v = row.get(col, 0)
        if col == "gi_absorption":
            v = {"High": 1.0, "Moderate": 0.6, "Low": 0.2}.get(str(v), 0.5)
        try:
            v = float(v or 0)
        except Exception:
            v = 0
        n = max(0.0, min(1.0, (v - lo) / (hi - lo + 1e-9)))
        return n if pos else 1.0 - n

    lbls = [p[1] for p in radar_props]
    sv   = [norm(start, p[0], p[2], p[3], p[4]) for p in radar_props]
    bv   = [norm(best,  p[0], p[2], p[3], p[4]) for p in radar_props]
    lbls += [lbls[0]]
    sv += [sv[0]]
    bv += [bv[0]]

    radar = go.Figure([
        go.Scatterpolar(r=sv, theta=lbls, fill="toself", name="Starting molecule", line=dict(color="#d97757", width=2), fillcolor="rgba(217,119,87,0.12)"),
        go.Scatterpolar(r=bv, theta=lbls, fill="toself", name=f"Best candidate (attempt {best['mol_index']})", line=dict(color="#0f766e", width=2), fillcolor="rgba(15,118,110,0.14)"),
    ])
    radar.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], tickfont=dict(size=9, color="#8c99a5"), gridcolor="#d8cdc0"),
            angularaxis=dict(tickfont=dict(size=11, color="#32404d")),
            bgcolor="#fffdf9",
        ),
        paper_bgcolor="#f7f0e8",
        legend=dict(font=dict(size=12), bgcolor="#fffdf9", bordercolor="#d8cdc0", borderwidth=1),
        margin=dict(t=30, b=30),
        height=400,
    )
    st.plotly_chart(
        radar,
        use_container_width=True,
        key=f"radar_chart_{render_token}",
    )

    smi = best.get("canonical_smiles") or best.get("input_smiles", "")
    st.markdown(
        f"""
<div class="best-banner">
  <div class="best-banner-label">Best candidate</div>
  <div class="best-banner-title">Attempt {best['mol_index']} is the clearest winner from this run.</div>
  <p class="best-banner-copy">Best overall balance in this run.</p>
  <div class="best-banner-code">{escape_html(smi)}</div>
  <div class="result-pill-row">
    <span class="result-pill">BBB {best.get('bbb_probability', 0) * 100:.0f}%</span>
    <span class="result-pill">CNS MPO {best.get('cns_mpo_score', 0):.2f}/6</span>
    <span class="result-pill">QED {best.get('qed_score', 0) * 100:.0f}%</span>
    <span class="result-pill">Safety alerts {best.get('num_alerts', 0)}</span>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_candidate_journey_tab(candidates):
    if not candidates:
        st.info("Run the optimisation to see molecule cards here.")
        return

    df = pd.DataFrame(candidates)
    start = df.iloc[0].to_dict()
    best_idx = int(df["cns_mpo_score"].idxmax()) if "cns_mpo_score" in df.columns else len(df) - 1

    render_section_heading(
        "Every molecule the agent evaluated",
        "Scan the full candidate path.",
        "Candidate journey",
    )
    st.write("")

    for i, c in enumerate(candidates):
        is_best = (i == best_idx)
        is_start = (i == 0)
        card_cls = "best" if is_best else ("start" if is_start else "good")
        previous_candidate = candidates[i - 1] if i > 0 else None

        if is_best:
            badge_cls, badge_txt = "badge-best", "⭐ Best Candidate"
        elif is_start:
            badge_cls, badge_txt = "badge-start", "Starting Molecule"
        else:
            badge_cls, badge_txt = "badge-attempt", f"Attempt {c['mol_index']}"

        smi = c.get("canonical_smiles") or c.get("input_smiles", "")

        st.markdown(f"""
<div class="mol-card {card_cls}">
  <span class="round-badge {badge_cls}">{badge_txt}</span>
""", unsafe_allow_html=True)

        col_img, col_scores = st.columns([1, 1], gap="large")

        with col_img:
            prev_smi = None
            if previous_candidate is not None:
                prev_smi = previous_candidate.get("canonical_smiles") or previous_candidate.get("input_smiles", "")
            svg_markup, has_highlight = mol_svg_markup(smi, prev_smi)
            if svg_markup:
                st.markdown(
                    f'<div class="mol-structure-panel">{svg_markup}</div>',
                    unsafe_allow_html=True,
                )
                if has_highlight:
                    st.markdown(
                        '<div class="change-note">Yellow highlight = region changed in this attempt</div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.markdown(
                    f'<div class="mol-structure-fallback">{escape_html(smi)}</div>',
                    unsafe_allow_html=True,
                )

        with col_scores:
            st.markdown(
                mol_scores_html(c, start if not is_start else None),
                unsafe_allow_html=True,
            )
            mw = c.get("molecular_weight", "?")
            logp = c.get("clogp", "?")
            st.caption(f"Mol. weight: **{mw} Da**  ·  Lipophilicity (cLogP): **{logp}**")

        reasoning = (c.get("reasoning") or "").strip()
        reasoning_summary = summarize_candidate_change(c, previous_candidate)
        if reasoning or reasoning_summary:
            section = "Initial take" if is_start else "Change summary"
            st.markdown(f"""
<div class="reasoning-box">
  <div class="reasoning-title">{escape_html(section)}</div>
  <div class="reasoning-text">{escape_html(reasoning_summary)}</div>
</div>
""", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)
        st.write("")


def render_tab_views(results_placeholder, journey_placeholder, candidates):
    st.session_state.render_seq += 1
    render_token = st.session_state.render_seq
    with journey_placeholder.container():
        render_candidate_journey_tab(candidates)
    with results_placeholder.container():
        render_results_tab(candidates, render_token=render_token)


# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Lead Optimizer")
    st.markdown(
        '<p class="page-subtitle">Define the brief, run once, compare fast.</p>',
        unsafe_allow_html=True,
    )
    st.divider()

    api_key = st.text_input(
        "Anthropic API Key",
        type="password",
        value=os.environ.get("ANTHROPIC_API_KEY", ""),
        help="Get yours free at console.anthropic.com",
    )

    st.divider()
    preset_name = st.selectbox("Example scenario", list(PRESETS.keys()))
    preset = PRESETS[preset_name]

    smiles_input = st.text_input(
        "Starting molecule (SMILES)",
        value=preset["smiles"],
        placeholder="Paste SMILES here…",
    )

    goal_input = st.text_area(
        "Target profile",
        value=preset["goal"],
        height=220,
        placeholder="Describe the profile you want to optimize toward…",
    )

    max_calls = st.slider("Max optimisation steps", 6, 30, 20, 2)

    st.divider()
    run_btn = st.button("Run Optimisation", type="primary", use_container_width=True)
    clear_btn = st.button("Clear results", use_container_width=True)

    if clear_btn:
        st.session_state.candidates = []
        st.session_state.completed  = False
        st.session_state.run_meta = {}
        st.session_state.run_notice = ""
        st.rerun()

    st.divider()
    st.markdown("#### Saved Runs")
    latest_exists = LATEST_RUN_PATH.exists()
    if st.button("Load latest saved run", use_container_width=True, disabled=not latest_exists):
        try:
            apply_loaded_run(json.loads(LATEST_RUN_PATH.read_text(encoding="utf-8")), "latest saved run")
            st.rerun()
        except Exception as exc:
            st.error(f"Could not load latest run: {exc}")

    uploaded_run = st.file_uploader(
        "Upload a saved run",
        type=["json"],
        help="Load previously saved results and view them in the UI without calling the API again.",
    )
    if uploaded_run is not None and st.button("Open uploaded run", use_container_width=True):
        try:
            apply_loaded_run(json.loads(uploaded_run.getvalue().decode("utf-8")), uploaded_run.name)
            st.rerun()
        except Exception as exc:
            st.error(f"Could not open uploaded run: {exc}")

    current_candidates = st.session_state.candidates
    if current_candidates:
        current_meta = st.session_state.get("run_meta") or {}
        download_payload = build_run_payload(
            current_meta.get("starting_smiles", smiles_input.strip()),
            current_meta.get("goal", goal_input.strip()),
            current_candidates,
            current_meta.get("preset_name", preset_name),
            source=current_meta.get("source", "live"),
        )
        st.download_button(
            "Download current run",
            data=json.dumps(download_payload, indent=2),
            file_name="lead_optimization_run.json",
            mime="application/json",
            use_container_width=True,
        )

    st.divider()
    st.caption("Claude agent loop with live scoring, plus local saved-run viewing")

current_meta = st.session_state.get("run_meta") or {}
smiles_clean = smiles_input.strip()
goal_clean = goal_input.strip()
active_smiles = current_meta.get("starting_smiles") or smiles_clean
active_goal = current_meta.get("goal") or goal_clean
active_preset_name = current_meta.get("preset_name") or preset_name
active_brief_title = brief_title_from_name(active_preset_name)
smiles_validation = is_valid_smiles(active_smiles) if active_smiles else {"valid": False}
candidates = st.session_state.candidates

if candidates:
    candidate_df = pd.DataFrame(candidates)
    best_candidate = (
        candidate_df.loc[candidate_df["cns_mpo_score"].idxmax()].to_dict()
        if "cns_mpo_score" in candidate_df.columns
        else candidate_df.iloc[-1].to_dict()
    )
    hero_stats = [
        ("Candidates explored", str(len(candidates))),
        ("Best CNS MPO", f"{best_candidate.get('cns_mpo_score', 0):.2f}/5"),
        ("Best BBB", f"{best_candidate.get('bbb_probability', 0) * 100:.0f}%"),
    ]
else:
    hero_stats = [
        ("Scenario", active_preset_name),
        ("Max steps", str(max_calls)),
        ("Run mode", "Guided optimisation loop"),
    ]

# ─── Header ────────────────────────────────────────────────────────────────────
hero_stats_html = "".join(
    f"""
<div class="hero-stat">
  <div class="hero-stat-label">{escape_html(label)}</div>
  <div class="hero-stat-value">{escape_html(value)}</div>
</div>
"""
    for label, value in hero_stats
)

st.markdown(
    f"""
<section class="hero-shell">
  <div class="hero-grid">
    <div>
      <span class="hero-eyebrow">Medicinal chemistry cockpit</span>
      <h1 class="hero-title">Lead optimisation,<span class="accent">designed for decision-making</span></h1>
      <p class="hero-text">
        A cleaner run view for quickly seeing whether a candidate is actually getting better.
      </p>
    </div>
    <div class="hero-panel">
      <div class="hero-panel-title">At a glance</div>
      <div class="hero-stat-grid">{hero_stats_html}</div>
    </div>
  </div>
</section>
""",
    unsafe_allow_html=True,
)

status_box = st.empty()
if st.session_state.run_notice:
    status_box.info(st.session_state.run_notice)
    st.session_state.run_notice = ""

overview_col, readiness_col = st.columns([1.4, 1], gap="large")

with overview_col:
    st.markdown(
        f"""
<div class="overview-card">
  <div class="overview-label">Current brief</div>
  <div class="overview-title">{escape_html(active_brief_title)}</div>
  {brief_body_html(active_goal)}
  <div class="brief-meta">Starting SMILES</div>
  <div class="mono-preview">{escape_html(active_smiles or 'No starting SMILES yet')}</div>
</div>
""",
        unsafe_allow_html=True,
    )

with readiness_col:
    readiness_html = "".join(
        [
            readiness_item(
                "API access",
                "Key is loaded."
                if api_key
                else "Add your Anthropic key.",
                bool(api_key),
            ),
            readiness_item(
                "Starting scaffold",
                "SMILES is valid."
                if smiles_validation.get("valid")
                else "Paste a valid SMILES.",
                bool(smiles_clean) and bool(smiles_validation.get("valid")),
            ),
            readiness_item(
                "Target brief",
                "Goal is clear."
                if goal_clean
                else "Add the target profile.",
                bool(goal_clean),
            ),
        ]
    )
    st.markdown(
        f"""
<div class="overview-card">
  <div class="overview-label">Run readiness</div>
  <div class="overview-title">Ready to run</div>
  <div class="readiness-list">{readiness_html}</div>
</div>
""",
        unsafe_allow_html=True,
    )

if not candidates:
    render_section_heading(
        "A lighter workflow",
        "Less explanation, more signal.",
        "Before you run",
    )
    st.markdown(
        """
<div class="empty-grid">
  <div class="empty-card">
    <div class="empty-step">Step 1</div>
    <div class="empty-title">Set the brief</div>
    <p class="empty-copy">Keep the task clear before you spend tokens.</p>
  </div>
  <div class="empty-card">
    <div class="empty-step">Step 2</div>
    <div class="empty-title">Run once</div>
    <p class="empty-copy">Catch missing inputs early.</p>
  </div>
  <div class="empty-card">
    <div class="empty-step">Step 3</div>
    <div class="empty-title">Compare outcomes</div>
    <p class="empty-copy">See the winner faster.</p>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

# ─── Tabs ──────────────────────────────────────────────────────────────────────
tab_molecules, tab_results = st.tabs(["Candidate Journey", "Performance Overview"])
with tab_molecules:
    journey_placeholder = st.empty()
with tab_results:
    results_placeholder = st.empty()

render_tab_views(results_placeholder, journey_placeholder, candidates)


# ─── Run ───────────────────────────────────────────────────────────────────────
if run_btn:
    if not api_key:
        st.error("Please enter your Anthropic API key in the sidebar.")
    elif not smiles_input.strip():
        st.error("Please enter a starting molecule SMILES.")
    elif not goal_input.strip():
        st.error("Please describe what you want to improve.")
    else:
        val = is_valid_smiles(smiles_input.strip())
        if not val["valid"]:
            st.error(f"Invalid SMILES: {val.get('error','could not be parsed')}")
        else:
            st.session_state.candidates = []
            st.session_state.completed  = False
            st.session_state.run_meta = {}
            render_tab_views(results_placeholder, journey_placeholder, [])
            with st.spinner("Agent running… scoring locally, no warm-up needed"):
                run_agent(
                    smiles_input.strip(),
                    goal_input.strip(),
                    max_calls,
                    api_key,
                    status_box,
                    results_placeholder=results_placeholder,
                    journey_placeholder=journey_placeholder,
                )
            if st.session_state.candidates:
                saved_payload = build_run_payload(
                    smiles_input.strip(),
                    goal_input.strip(),
                    st.session_state.candidates,
                    preset_name,
                    source="live",
                )
                try:
                    save_run_snapshot(saved_payload, RUNS_DIR, LATEST_RUN_PATH)
                    st.session_state.run_notice = "Run saved locally. You can reopen it later without calling the API again."
                except Exception as exc:
                    st.session_state.run_notice = f"Run finished, but local save failed: {exc}"
                st.session_state.run_meta = {
                    "preset_name": preset_name,
                    "starting_smiles": smiles_input.strip(),
                    "goal": goal_input.strip(),
                    "saved_at": saved_payload["saved_at"],
                    "source": "live",
                    "source_label": "latest live run",
                }
            st.rerun()
