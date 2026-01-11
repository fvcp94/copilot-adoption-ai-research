from __future__ import annotations

# ============================================================
# Streamlit Cloud import fix (REQUIRED for pages)
# ============================================================
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # repo root
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# ============================================================

import pandas as pd
import streamlit as st
import plotly.express as px

from src.io import load_data
from src.metrics import adoption_rate, attention_pass_rate

st.set_page_config(page_title="Insights & Recommendations", layout="wide")
st.title("Insights & Recommendations")
st.caption("Executive-friendly summary: what we found, what to do, and caveats.")

df = load_data()

# ------------------------------------------------------------
# Column resolver
# ------------------------------------------------------------
def pick_col(*names: str) -> str | None:
    for n in names:
        if n in df.columns:
            return n
    return None

adopt_col = pick_col("uses_copilot", "copilot_adopted")
sat_col = pick_col("satisfaction_1to5", "satisfaction")
prod_col = pick_col("productivity_gain_pct", "productivity_score", "productivity_gain")
attn_col = pick_col("attention_check_pass", "attention_check_passed")

# ------------------------------------------------------------
# KPIs
# ------------------------------------------------------------
st.header("📌 Headline Metrics")

ar = adoption_rate(df)
apr = attention_pass_rate(df)

c1, c2, c3 = st.columns(3)

with c1:
    st.metric("Adoption Rate", f"{ar:.1%}")
with c2:
    st.metric("Attention Pass Rate", f"{apr:.1%}")
with c3:
    st.metric("Sample Size", f"{len(df):,}")

# ------------------------------------------------------------
# Key Findings (auto-computed)
# ------------------------------------------------------------
st.header("✅ Key Findings (Data-driven)")

findings = []

# Adoption segments
if adopt_col:
    adopt_by_role = df.groupby("role")[adopt_col].mean().sort_values(ascending=False)
    top_role = adopt_by_role.index[0]
    low_role = adopt_by_role.index[-1]
    findings.append(
        f"Adoption varies by **role**: highest in **{top_role}** ({adopt_by_role.iloc[0]:.1%}) "
        f"and lowest in **{low_role}** ({adopt_by_role.iloc[-1]:.1%})."
    )

    adopt_by_region = df.groupby("region")[adopt_col].mean().sort_values(ascending=False)
    top_region = adopt_by_region.index[0]
    low_region = adopt_by_region.index[-1]
    findings.append(
        f"Adoption varies by **region**: highest in **{top_region}** ({adopt_by_region.iloc[0]:.1%}) "
        f"and lowest in **{low_region}** ({adopt_by_region.iloc[-1]:.1%})."
    )

# Satisfaction difference
if adopt_col and sat_col:
    adopters = df.loc[df[adopt_col].astype(int) == 1, sat_col]
    non = df.loc[df[adopt_col].astype(int) == 0, sat_col]
    if len(adopters.dropna()) > 10 and len(non.dropna()) > 10:
        diff = adopters.mean() - non.mean()
        findings.append(
            f"Average **satisfaction** is higher among adopters by **{diff:.2f} points** on a 1–5 scale."
        )

# Productivity difference
if adopt_col and prod_col:
    adopters = pd.to_numeric(df.loc[df[adopt_col].astype(int) == 1, prod_col], errors="coerce")
    non = pd.to_numeric(df.loc[df[adopt_col].astype(int) == 0, prod_col], errors="coerce")
    if len(adopters.dropna()) > 10 and len(non.dropna()) > 10:
        diff = adopters.mean() - non.mean()
        findings.append(
            f"Estimated **productivity gain** is higher for adopters by **{diff:.2f}** (units depend on metric; here it is % if using `productivity_gain_pct`)."
        )

if not findings:
    st.info("Not enough columns found to compute automated insights. (This is okay for a demo.)")
else:
    for f in findings:
        st.write("• " + f)

# ------------------------------------------------------------
# Recommended Actions
# ------------------------------------------------------------
st.header("🎯 Recommendations (What to do next)")

st.markdown(
    """
**1) Target onboarding and enablement where adoption is lowest**  
Focus training, templates, and in-product guidance on the segments with the lowest adoption.

**2) Improve perceived value with role-specific use cases**  
Create curated workflows for ICs vs managers (emails, meeting summaries, document drafts, etc.).

**3) Use experiments to validate improvements**  
Run onboarding and messaging A/B tests to measure lift in adoption and satisfaction.

**4) Monitor quality and bias**  
Use attention checks, weighting, and sensitivity analyses to avoid biased conclusions.
"""
)

# ------------------------------------------------------------
# Visual summaries
# ------------------------------------------------------------
st.header("📈 Visual Summary")

if adopt_col:
    col1, col2 = st.columns(2)

    with col1:
        adopt_by_role = (
            df.groupby("role")[adopt_col].mean().sort_values(ascending=False).reset_index()
            .rename(columns={adopt_col: "adoption_rate"})
        )
        st.plotly_chart(
            px.bar(adopt_by_role, x="role", y="adoption_rate", title="Adoption rate by role"),
            use_container_width=True,
        )

    with col2:
        adopt_by_region = (
            df.groupby("region")[adopt_col].mean().sort_values(ascending=False).reset_index()
            .rename(columns={adopt_col: "adoption_rate"})
        )
        st.plotly_chart(
            px.bar(adopt_by_region, x="region", y="adoption_rate", title="Adoption rate by region"),
            use_container_width=True,
        )

# ------------------------------------------------------------
# Caveats
# ------------------------------------------------------------
st.header("⚠️ Caveats")

st.markdown(
    """
- This is a **synthetic dataset** for demonstration and reproducibility.
- Survey metrics can suffer from **self-report bias** and **non-response bias**.
- Causal claims require experiments or careful causal inference with strong assumptions.
"""
)

st.success("Insights page loaded successfully.")
