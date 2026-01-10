from __future__ import annotations

# ============================================================
# Streamlit Cloud import fix (REQUIRED)
# ============================================================
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# ============================================================

import subprocess
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# Project imports (now work correctly)
from src.io import load_data, load_codebook
from src.metrics import adoption_rate, attention_pass_rate

# Load environment variables (OpenRouter, etc.)
load_dotenv()

# ============================================================
# Page configuration
# ============================================================
st.set_page_config(
    page_title="Copilot Adoption & Satisfaction Analytics",
    layout="wide",
)

# ============================================================
# Sidebar
# ============================================================
st.sidebar.title("📊 Copilot Research Platform")
st.sidebar.markdown(
    """
    **Survey Research & AI Analytics**
    
    - Adoption & engagement metrics  
    - Survey quality diagnostics  
    - Inferential & causal analysis  
    - AI-assisted research tooling  
    """
)

# ============================================================
# Load data
# ============================================================
@st.cache_data(show_spinner=True)
def get_data():
    data = load_data()
    codebook = load_codebook()
    return data, codebook


data, codebook = get_data()

# ============================================================
# Main Dashboard
# ============================================================
st.title("Copilot Adoption & Satisfaction Analytics")

st.markdown(
    """
    This dashboard demonstrates **rigorous survey research methods**
    combined with **AI-enhanced analytics**, designed to support
    product and decision-making teams.
    """
)

# ============================================================
# KPI Section
# ============================================================
col1, col2 = st.columns(2)

with col1:
    st.metric(
        label="Adoption Rate",
        value=f"{adoption_rate(data):.1%}",
        help="Percentage of respondents actively using Copilot",
    )

with col2:
    st.metric(
        label="Attention Pass Rate",
        value=f"{attention_pass_rate(data):.1%}",
        help="Survey quality metric based on attention checks",
    )

# ============================================================
# Data Preview
# ============================================================
with st.expander("📄 Preview Survey Data"):
    st.dataframe(data.head(50), use_container_width=True)

# ============================================================
# Codebook
# ============================================================
with st.expander("📘 Survey Codebook"):
    st.dataframe(codebook, use_container_width=True)

# ============================================================
# Footer
# ============================================================
st.markdown("---")
st.caption(
    "Built with Python, Streamlit, statistical inference, and OpenRouter-powered LLMs "
    "(free-tier compatible)."
)
