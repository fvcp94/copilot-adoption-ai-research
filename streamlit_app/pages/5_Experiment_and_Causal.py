from __future__ import annotations

import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

from src.io import load_data

st.title("Experimental & Causal Analysis")
st.caption("Onboarding experiment + propensity score matching demo (synthetic).")

@st.cache_data
def get_df():
    return load_data()

df = get_df()

st.subheader("Onboarding experiment (eligible users only)")
eligible = df[df["eligible_onboarding"] == 1].copy()
if len(eligible) < 50:
    st.warning("Not enough eligible users to analyze.")
else:
    by = eligible.groupby("onboarding_treatment")["copilot_adopted"].mean().rename("adoption_rate").reset_index()
    st.dataframe(by, use_container_width=True)
    lift = float(by.loc[by.onboarding_treatment == 1, "adoption_rate"].values[0] - by.loc[by.onboarding_treatment == 0, "adoption_rate"].values[0])
    st.metric("Treatment - Control (adoption lift)", f"{lift:.4f}")

st.divider()
st.subheader("Propensity Score Matching (PSM): adoption impact on productivity")

work = df.copy().dropna(subset=["productivity_score", "m365_usage", "ai_openness"])
X = pd.get_dummies(work[["m365_usage", "ai_openness", "company_size", "job_family", "region", "tenure_bucket"]], drop_first=True)
t = work["copilot_adopted"].astype(int).to_numpy()

clf = LogisticRegression(max_iter=1000)
clf.fit(X, t)
work["propensity_score"] = clf.predict_proba(X)[:, 1]

treated = work[work["copilot_adopted"] == 1].copy()
control = work[work["copilot_adopted"] == 0].copy()

nn = NearestNeighbors(n_neighbors=1)
nn.fit(control[["propensity_score"]].to_numpy())
_, idx = nn.kneighbors(treated[["propensity_score"]].to_numpy())

matched_control = control.iloc[idx.flatten()].copy()
matched_control.index = treated.index

ate = (treated["productivity_score"] - matched_control["productivity_score"]).mean()
st.metric("PSM estimate (ATE) on productivity_score", f"{ate:.3f}")

with st.expander("Matched sample preview"):
    preview = pd.DataFrame({
        "treated_prod": treated["productivity_score"].head(20),
        "matched_control_prod": matched_control["productivity_score"].head(20),
        "treated_ps": treated["propensity_score"].head(20),
        "matched_control_ps": matched_control["propensity_score"].head(20),
    })
    st.dataframe(preview, use_container_width=True)
