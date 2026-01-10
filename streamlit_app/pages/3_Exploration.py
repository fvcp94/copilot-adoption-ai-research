from __future__ import annotations

import streamlit as st
from src.io import load_data
from src.viz import hist, bar_rate

st.title("Exploratory Analysis")
st.caption("Distributions and adoption patterns across segments.")

@st.cache_data
def get_df():
    return load_data()

df = get_df()

st.plotly_chart(hist(df, "m365_usage", "Microsoft 365 usage intensity"), use_container_width=True)
st.plotly_chart(hist(df, "ai_openness", "AI openness"), use_container_width=True)

st.divider()
st.subheader("Adoption by segment (unweighted)")
st.plotly_chart(bar_rate(df, "company_size", "copilot_adopted", "Adoption rate by company size"), use_container_width=True)
st.plotly_chart(bar_rate(df, "job_family", "copilot_adopted", "Adoption rate by job family"), use_container_width=True)
st.plotly_chart(bar_rate(df, "region", "copilot_adopted", "Adoption rate by region"), use_container_width=True)
