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

st.set_page_config(page_title="Exploratory Analysis", layout="wide")
st.title("Exploratory Analysis")
st.caption("Distributions and adoption patterns across segments.")

df = load_data()

# ------------------------------------------------------------
# Helpful column resolver (handles naming differences)
# ------------------------------------------------------------
def pick_col(*names: str) -> str | None:
    for n in names:
        if n in df.columns:
            return n
    return None

copilot_col = pick_col("uses_copilot", "copilot_adopted")
attn_col = pick_col("attention_check_pass", "attention_check_passed")

# ------------------------------------------------------------
# Distribution charts
# ------------------------------------------------------------
st.header("📊 Distributions")

c1, c2, c3 = st.columns(3)

with c1:
    st.plotly_chart(
        px.histogram(df, x="docs_per_week", nbins=40, title="Docs per week"),
        use_container_width=True,
    )

with c2:
    st.plotly_chart(
        px.histogram(df, x="weekly_meetings", nbins=40, title="Weekly meetings"),
        use_container_width=True,
    )

with c3:
    st.plotly_chart(
        px.histogram(df, x="tenure_months", nbins=40, title="Tenure (months)"),
        use_container_width=True,
    )

# ------------------------------------------------------------
# Adoption patterns
# ------------------------------------------------------------
st.header("🚀 Copilot Adoption Patterns")

if copilot_col is None:
    st.error("Missing adoption column. Expected one of: uses_copilot, copilot_adopted")
else:
    # Adoption by role
    col1, col2 = st.columns(2)

    with col1:
        adopt_by_role = (
            df.groupby("role")[copilot_col]
            .mean()
            .sort_values(ascending=False)
            .reset_index()
            .rename(columns={copilot_col: "adoption_rate"})
        )
        st.plotly_chart(
            px.bar(adopt_by_role, x="role", y="adoption_rate", title="Adoption rate by role"),
            use_container_width=True,
        )

    with col2:
        adopt_by_region = (
            df.groupby("region")[copilot_col]
            .mean()
            .sort_values(ascending=False)
            .reset_index()
            .rename(columns={copilot_col: "adoption_rate"})
        )
        st.plotly_chart(
            px.bar(adopt_by_region, x="region", y="adoption_rate", title="Adoption rate by region"),
            use_container_width=True,
        )

# ------------------------------------------------------------
# Satisfaction & productivity (if present)
# ------------------------------------------------------------
st.header("😊 Satisfaction & Productivity")

sat_col = pick_col("satisfaction_1to5", "satisfaction")
prod_col = pick_col("productivity_gain_pct", "productivity_gain")

col1, col2 = st.columns(2)

with col1:
    if sat_col:
        st.plotly_chart(
            px.box(df, x="role", y=sat_col, title="Satisfaction by role"),
            use_container_width=True,
        )
    else:
        st.info("Satisfaction column not found.")

with col2:
    if prod_col:
        st.plotly_chart(
            px.box(df, x="role", y=prod_col, title="Productivity gain (%) by role"),
            use_container_width=True,
        )
    else:
        st.info("Productivity column not found.")

# ------------------------------------------------------------
# Data quality segment check
# ------------------------------------------------------------
st.header("✅ Attention Check (Quality)")

if attn_col:
    pass_rate_by_role = (
        df.groupby("role")[attn_col]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={attn_col: "pass_rate"})
    )
    st.plotly_chart(
        px.bar(pass_rate_by_role, x="role", y="pass_rate", title="Attention pass rate by role"),
        use_container_width=True,
    )
else:
    st.info("Attention check column not found.")

st.success("Exploration page loaded successfully.")
