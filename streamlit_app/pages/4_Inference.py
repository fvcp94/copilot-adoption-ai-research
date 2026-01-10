from __future__ import annotations

import pandas as pd
import streamlit as st

from src.io import load_data
from src.stats_models import two_sample_ttest, adoption_logit, productivity_ols

st.title("Inferential Statistics")
st.caption("Hypothesis testing + regression models (with interpretation).")

@st.cache_data
def get_df():
    return load_data()

df = get_df()

st.subheader("Comparison: adopters vs non-adopters (productivity)")
res = two_sample_ttest(df, outcome="productivity_score", group="copilot_adopted")
st.dataframe(pd.DataFrame([res]), use_container_width=True)

st.markdown("This is association (not causality). See **Experiment & Causal** for matching-based estimate.")

st.divider()
st.subheader("Logistic regression: adoption drivers")
st.text(adoption_logit(df).summary().as_text())

st.divider()
st.subheader("OLS regression: productivity ~ adoption + controls")
st.text(productivity_ols(df).summary().as_text())
