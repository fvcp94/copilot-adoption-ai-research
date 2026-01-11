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

from src.io import load_data


st.set_page_config(page_title="Data Quality", layout="wide")
st.title("Data Quality")

st.markdown(
    """
This page demonstrates **survey data quality diagnostics** commonly used in product research:
- Missingness checks
- Attention check pass rate
- Simple straight-lining / low-variance indicators
"""
)

df = load_data()

# ------------------------------------------------------------
# Basic schema + preview
# ------------------------------------------------------------
with st.expander("Preview data"):
    st.dataframe(df.head(50), use_container_width=True)

st.header("✅ Quality Metrics")

col1, col2, col3 = st.columns(3)

# Attention check
if "attention_check_pass" in df.columns:
    attention_rate = df["attention_check_pass"].mean()
else:
    attention_rate = float("nan")

with col1:
    st.metric("Attention Pass Rate", f"{attention_rate:.1%}" if pd.notna(attention_rate) else "N/A")

# Missingness
missing_rate = df.isna().mean().mean()

with col2:
    st.metric("Overall Missingness", f"{missing_rate:.1%}")

# Duplicate respondent_id
dup_rate = 0.0
if "respondent_id" in df.columns:
    dup_rate = df["respondent_id"].duplicated().mean()

with col3:
    st.metric("Duplicate IDs", f"{dup_rate:.1%}")

# ------------------------------------------------------------
# Missingness by column
# ------------------------------------------------------------
st.header("📉 Missingness by Column")

missing_by_col = (
    df.isna()
    .mean()
    .sort_values(ascending=False)
    .reset_index()
    .rename(columns={"index": "column", 0: "missing_rate"})
)

st.dataframe(missing_by_col, use_container_width=True)

# ------------------------------------------------------------
# Straight-lining proxy (low variance) for numeric columns
# ------------------------------------------------------------
st.header("🧠 Straight-lining / Low-Variance Proxy")

numeric_cols = df.select_dtypes(include="number").columns.tolist()

if numeric_cols:
    variances = df[numeric_cols].var(numeric_only=True).sort_values()
    st.dataframe(
        variances.reset_index().rename(columns={"index": "column", 0: "variance"}),
        use_container_width=True,
    )
    st.caption(
        "Very low variance numeric fields can be a signal of low-quality responses "
        "(in real surveys you'd use item-level response patterns)."
    )
else:
    st.info("No numeric columns found for variance checks.")

st.success("Data Quality page loaded successfully.")
