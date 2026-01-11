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

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import NearestNeighbors

from src.io import load_data

st.set_page_config(page_title="Experimental & Causal", layout="wide")
st.title("Experimental & Causal Analysis")
st.caption("Onboarding experiment + propensity score matching demo (synthetic).")

df = load_data().copy()

# ------------------------------------------------------------
# Column resolver
# ------------------------------------------------------------
def pick_col(*names: str) -> str | None:
    for n in names:
        if n in df.columns:
            return n
    return None

adopt_col = pick_col("uses_copilot", "copilot_adopted")
outcome_col = pick_col("productivity_gain_pct", "productivity_score", "productivity_gain")

if adopt_col is None or outcome_col is None:
    st.error("Missing required columns for demo (adoption or outcome).")
    st.stop()

# Make sure adoption is 0/1 int
if df[adopt_col].dtype == bool:
    df["_adopt"] = df[adopt_col].astype(int)
else:
    df["_adopt"] = pd.to_numeric(df[adopt_col], errors="coerce").fillna(0).astype(int)

df["_y"] = pd.to_numeric(df[outcome_col], errors="coerce")

# ------------------------------------------------------------
# Create a synthetic "eligible onboarding" indicator
# (e.g., knowledge workers with certain license types + enough workload)
# ------------------------------------------------------------
st.header("Onboarding experiment (eligible users only)")

license_col = pick_col("license_type")
if license_col is None:
    # If missing, assume everyone eligible
    df["eligible_onboarding"] = 1
else:
    # Example eligibility rule: not already using copilot + has workload + license is M365 or Enterprise
    df["eligible_onboarding"] = (
        (df["_adopt"] == 0)
        & (df.get("docs_per_week", 0) >= 8)
        & (df.get("weekly_meetings", 0) >= 4)
        & (df[license_col].isin(["M365", "Enterprise", "M365+Copilot"]))
    ).astype(int)

eligible = df[df["eligible_onboarding"] == 1].copy()

st.write(f"Eligible sample size: **{len(eligible):,}**")

if len(eligible) < 200:
    st.warning("Eligible sample is small; results will be noisy (synthetic demo).")

# ------------------------------------------------------------
# Randomize onboarding experiment (synthetic)
# Treatment: improved onboarding flow vs control
# Outcome: adoption within 30 days (synthetic)
# ------------------------------------------------------------
rng = np.random.default_rng(42)

eligible["treatment"] = rng.binomial(1, 0.5, size=len(eligible))

# Synthetic adoption-after-onboarding probability
# baseline depends on workload + role + treatment boost
base = 0.18
base += np.clip((eligible.get("docs_per_week", 0) - 12) / 80, -0.05, 0.10)
base += np.where(eligible.get("role", "") == "Manager", 0.04, 0.0)
base += np.where(eligible.get("role", "") == "Director", 0.06, 0.0)
base += 0.08 * eligible["treatment"]  # onboarding improvement lifts adoption
p_post = np.clip(base, 0.02, 0.85)

eligible["adopted_post_30d"] = rng.binomial(1, p_post, size=len(eligible))

# Effect estimate
p_t = eligible.loc[eligible["treatment"] == 1, "adopted_post_30d"].mean()
p_c = eligible.loc[eligible["treatment"] == 0, "adopted_post_30d"].mean()
lift = p_t - p_c

st.subheader("A/B Result (Adoption within 30 days)")
c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Treatment adoption", f"{p_t:.1%}")
with c2:
    st.metric("Control adoption", f"{p_c:.1%}")
with c3:
    st.metric("Lift (pp)", f"{lift*100:.2f}")

# Simple 2-proportion z-test (approx)
n_t = (eligible["treatment"] == 1).sum()
n_c = (eligible["treatment"] == 0).sum()
p_pool = (p_t * n_t + p_c * n_c) / (n_t + n_c)
se = np.sqrt(p_pool * (1 - p_pool) * (1 / n_t + 1 / n_c))
z = (p_t - p_c) / se if se > 0 else np.nan
p_val = 2 * (1 - stats.norm.cdf(abs(z))) if np.isfinite(z) else np.nan

st.write(f"Approx z-test p-value: **{p_val:.4f}**")

st.plotly_chart(
    px.bar(
        pd.DataFrame(
            {"group": ["Control", "Treatment"], "adoption_rate": [p_c, p_t]}
        ),
        x="group",
        y="adoption_rate",
        title="Adoption within 30 days by onboarding group",
    ),
    use_container_width=True,
)

st.markdown(
    """
**Interpretation (plain English):**
- We simulated an onboarding experiment for eligible users.
- Treatment users receive an improved onboarding experience.
- The lift estimates how much onboarding improves adoption.
"""
)

# ------------------------------------------------------------
# Propensity Score Matching (PSM) demo
# Goal: estimate adoption effect on productivity, adjusting for confounders
# ------------------------------------------------------------
st.header("Propensity Score Matching (PSM) demo")

st.markdown(
    """
Here we demonstrate causal adjustment using PSM:

- **Treatment**: uses Copilot (adoption)
- **Outcome**: productivity gain (%)
- **Covariates**: role, region, device, license, workload, tenure
"""
)

psm_df = df.dropna(subset=["_y"]).copy()

covariates = []
cat_cols = []
num_cols = []

for c in ["role", "region", "device", "license_type"]:
    if c in psm_df.columns:
        covariates.append(c)
        cat_cols.append(c)

for c in ["tenure_months", "weekly_meetings", "docs_per_week"]:
    if c in psm_df.columns:
        covariates.append(c)
        num_cols.append(c)

if len(covariates) == 0:
    st.warning("No covariates found; PSM demo cannot run.")
    st.stop()

X = psm_df[covariates]
y = psm_df["_adopt"]

preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", "passthrough", num_cols),
    ]
)

model = Pipeline(
    steps=[
        ("prep", preprocess),
        ("lr", LogisticRegression(max_iter=200)),
    ]
)

model.fit(X, y)
ps = model.predict_proba(X)[:, 1]
psm_df["propensity_score"] = ps

# Nearest neighbor matching on propensity score
treated = psm_df[psm_df["_adopt"] == 1].copy()
control = psm_df[psm_df["_adopt"] == 0].copy()

if len(treated) < 50 or len(control) < 50:
    st.warning("Not enough treated/control samples for stable matching (synthetic).")
    st.stop()

nn = NearestNeighbors(n_neighbors=1)
nn.fit(control[["propensity_score"]])
dist, idx = nn.kneighbors(treated[["propensity_score"]])

matched_control = control.iloc[idx.flatten()].copy()
matched_control.index = treated.index  # align

ate = treated["_y"].mean() - matched_control["_y"].mean()

st.subheader("PSM Effect Estimate (Adoption → Productivity)")
st.metric("Estimated ATE (mean difference)", f"{ate:.2f}")

st.plotly_chart(
    px.histogram(
        psm_df,
        x="propensity_score",
        color=psm_df["_adopt"].map({0: "Non-adopter", 1: "Adopter"}),
        nbins=40,
        title="Propensity score distribution (adopters vs non-adopters)",
    ),
    use_container_width=True,
)

st.markdown(
    """
**Interpretation:**
- Propensity scores estimate the probability of adoption given observed covariates.
- We match adopters to similar non-adopters and compare productivity.
- This reduces bias from measured confounders (still synthetic, demo-only).
"""
)

st.success("Experimental & Causal page loaded successfully.")
