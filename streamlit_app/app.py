from __future__ import annotations
import streamlit_app

import sys
import subprocess
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from src.io import load_data, load_codebook
from src.metrics import adoption_rate, attention_pass_rate

load_dotenv()

st.set_page_config(page_title="Copilot Adoption & Satisfaction Analytics", layout="wide")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "synthetic_survey_data.csv"


def ensure_data_exists():
    if DATA_PATH.exists():
        return
    # Generate synthetic data on first run (works on Streamlit Cloud)
    cmd = [sys.executable, str(PROJECT_ROOT / "data_generation" / "data_generation_process.py")]
    subprocess.check_call(cmd)


@st.cache_data
def get_df():
    ensure_data_exists()
    return load_data()


df = get_df()
codebook = load_codebook()

st.title("Copilot Adoption & Satisfaction Analytics Platform")
st.caption("Survey research rigor + AI-enhanced survey tooling (synthetic demo).")

st.sidebar.header("Filters")
region = st.sidebar.multiselect("Region", sorted(df["region"].dropna().unique()), default=sorted(df["region"].dropna().unique()))
company = st.sidebar.multiselect("Company size", sorted(df["company_size"].dropna().unique()), default=sorted(df["company_size"].dropna().unique()))
job = st.sidebar.multiselect("Job family", sorted(df["job_family"].dropna().unique()), default=sorted(df["job_family"].dropna().unique()))

f = df[df["region"].isin(region) & df["company_size"].isin(company) & df["job_family"].isin(job)].copy()

uw, ww = adoption_rate(f)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Respondents (filtered)", f"{len(f):,}")
c2.metric("Adoption rate (unweighted)", f"{uw:.3f}")
c3.metric("Adoption rate (weighted)", f"{ww:.3f}" if pd.notna(ww) else "NA")
c4.metric("Attention pass rate", f"{attention_pass_rate(f):.3f}" if pd.notna(attention_pass_rate(f)) else "NA")

st.divider()

st.subheader("How to use this dashboard")
st.markdown(
"""
- Use the **filters** to slice by region/company/job family.
- Use the **Pages** menu on the left to navigate:
  - Research Design
  - Data Quality
  - Exploration
  - Inference
  - Experiment & Causal
  - Insights
  - AI Question Generator (OpenRouter)
"""
)

with st.expander("Data dictionary (codebook)"):
    if codebook:
        st.json(codebook)
    else:
        st.info("codebook.json not found (it will be created during data generation).")
