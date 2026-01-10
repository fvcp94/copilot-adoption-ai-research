from __future__ import annotations

import pandas as pd
import streamlit as st

from src.io import load_data
from src.metrics import completion_rate, attention_pass_rate

st.title("Data Quality")
st.caption("Response quality checks, missingness, and survey hygiene metrics.")

@st.cache_data
def get_df():
    return load_data()

df = get_df()

c1, c2, c3 = st.columns(3)
c1.metric("Completion rate", f"{completion_rate(df):.3f}" if pd.notna(completion_rate(df)) else "NA")
c2.metric("Attention pass rate", f"{attention_pass_rate(df):.3f}" if pd.notna(attention_pass_rate(df)) else "NA")
c3.metric("Straight-lining flagged", f"{(df['straight_lining_flag']==1).mean():.3f}" if "straight_lining_flag" in df.columns else "NA")

st.divider()
st.subheader("Missingness snapshot")
miss = df.isna().mean().sort_values(ascending=False).rename("missing_rate").reset_index().rename(columns={"index": "column"})
st.dataframe(miss.head(20), use_container_width=True)
