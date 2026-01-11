from __future__ import annotations

import streamlit as st
import pandas as pd

# ------------------------------------------------------------
# Robust imports (Cloud-safe)
# ------------------------------------------------------------
from src.io import (
    load_data,
    load_codebook,
    load_population_margins,
)

# ------------------------------------------------------------
# Page config
# ------------------------------------------------------------
st.set_page_config(
    page_title="Research Design",
    layout="wide",
)

st.title("Research Design")

st.markdown(
    """
This section documents the **survey research methodology** used to study
Copilot adoption and satisfaction.

The goal is to demonstrate **statistical rigor**, transparency,
and best practices in survey-based product research.
"""
)

# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------
data = load_data()
codebook = load_codebook()
population_margins = load_population_margins()

# ------------------------------------------------------------
# Survey Overview
# ------------------------------------------------------------
st.header("📋 Survey Overview")

col1, col2 = st.columns(2)

with col1:
    st.markdown(
        """
**Target Population**
- Knowledge workers using Microsoft 365
- Mix of individual contributors and managers
- Multiple regions and device ecosystems
"""
    )

with col2:
    st.markdown(
        f"""
**Sample Characteristics**
- Sample size: **{len(data):,} respondents**
- Synthetic but distributionally realistic
- Designed for reproducible experimentation
"""
    )

# ------------------------------------------------------------
# Sampling Strategy
# ------------------------------------------------------------
st.header("🎯 Sampling Strategy")

st.markdown(
    """
A **stratified sampling approach** is assumed to ensure representation across
key dimensions that influence Copilot usage:

- Role / seniority
- Geographic region
- Device ecosystem
- License type
"""
)

st.subheader("Population Margins (Target Distributions)")

for dim, margins in population_margins.items():
    st.markdown(f"**{dim.title()}**")
    st.dataframe(
        pd.DataFrame(
            {
                "Category": margins.keys(),
                "Target Proportion": margins.values(),
            }
        ),
        use_container_width=True,
    )

# ------------------------------------------------------------
# Questionnaire Design
# ------------------------------------------------------------
st.header("📝 Questionnaire Design")

st.markdown(
    """
The questionnaire combines:

- **Behavioral measures** (Copilot usage)
- **Attitudinal measures** (satisfaction)
- **Outcome measures** (productivity gains)
- **Data quality checks** (attention checks)
"""
)

with st.expander("View Survey Variables"):
    st.dataframe(codebook, use_container_width=True)

# ------------------------------------------------------------
# Data Quality Controls
# ------------------------------------------------------------
st.header("✅ Data Quality Controls")

attention_pass_rate = data["attention_check_pass"].mean()

st.markdown(
    f"""
To ensure reliability of insights:

- Attention checks were embedded in the survey
- **Attention pass rate:** **{attention_pass_rate:.1%}**
- Responses failing quality checks can be excluded downstream
"""
)

# ------------------------------------------------------------
# Assumptions & Limitations
# ------------------------------------------------------------
st.header("⚠️ Assumptions & Limitations")

st.markdown(
    """
**Key Assumptions**
- Responses are independent
- Self-reported productivity is directionally accurate
- Synthetic data approximates real-world distributions

**Limitations**
- Cross-sectional design (no causal claims without experiments)
- Self-report bias
- Synthetic dataset used for demonstration purposes
"""
)

# ------------------------------------------------------------
# Reprod
