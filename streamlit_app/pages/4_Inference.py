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
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

from src.io import load_data


st.set_page_config(page_title="Inferential Statistics", layout="wide")
st.title("Inferential Statistics")
st.caption("Hypothesis testing + regression models (with interpretation).")

df = load_data()

# ------------------------------------------------------------
# Column resolver
# ------------------------------------------------------------
def pick_col(*names: str) -> str | None:
    for n in names:
        if n in df.columns:
            return n
    return None

group_col = pick_col("uses_copilot", "copilot_adopted")
outcome_col = pick_col("productivity_gain_pct", "productivity_score", "productivity_gain")

if group_col is None:
    st.error("Missing adoption group column. Expected one of: uses_copilot, copilot_adopted")
    st.stop()

if outcome_col is None:
    st.error("Missing productivity outcome column. Expected one of: productivity_gain_pct, productivity_score, productivity_gain")
    st.stop()

# Ensure boolean/int grouping
g = df[group_col]
if g.dtype == bool:
    df["_grp"] = g.astype(int)
else:
    df["_grp"] = pd.to_numeric(g, errors="coerce").fillna(0).astype(int)

df["_y"] = pd.to_numeric(df[outcome_col], errors="coerce")

st.header("Comparison: adopters vs non-adopters (productivity)")

adopters = df.loc[df["_grp"] == 1, "_y"].dropna()
non_adopters = df.loc[df["_grp"] == 0, "_y"].dropna()

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Adopters (n)", f"{len(adopters):,}")
with col2:
    st.metric("Non-adopters (n)", f"{len(non_adopters):,}")
with col3:
    st.metric("Outcome Column", outcome_col)

# Plot distribution
st.plotly_chart(
    px.box(
        df.dropna(subset=["_y"]),
        x="_grp",
        y="_y",
        points="outliers",
        labels={"_grp": "Uses Copilot (0/1)", "_y": outcome_col},
        title="Productivity distribution by adoption group",
    ),
    use_container_width=True,
)

# Welch’s t-test
t_stat, p_val = stats.ttest_ind(adopters, non_adopters, equal_var=False)

# Effect size (Cohen's d with pooled SD)
mean_a, mean_b = adopters.mean(), non_adopters.mean()
sd_a, sd_b = adopters.std(ddof=1), non_adopters.std(ddof=1)
pooled = ((sd_a**2 + sd_b**2) / 2) ** 0.5
cohens_d = (mean_a - mean_b) / pooled if pooled > 0 else float("nan")

st.subheader("Welch Two-Sample t-test")

c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Mean (adopters)", f"{mean_a:.2f}")
with c2:
    st.metric("Mean (non-adopters)", f"{mean_b:.2f}")
with c3:
    st.metric("p-value", f"{p_val:.4f}")

st.markdown(
    f"""
**Interpretation (plain English):**
- We compare the average **{outcome_col}** between Copilot adopters vs non-adopters.
- The test uses **Welch’s t-test**, which does not assume equal variances.
- **Effect size (Cohen’s d):** `{cohens_d:.2f}` (magnitude of difference in SD units).
"""
)

# ------------------------------------------------------------
# Regression: drivers of productivity
# ------------------------------------------------------------
st.header("Regression: productivity drivers")

st.markdown(
    """
We fit a simple linear regression with:
- Adoption (uses_copilot)
- Role, region, device, license_type
- Tenure and workload proxies (meetings, docs)
"""
)

# Only keep columns that exist
base_cols = ["role", "region", "device", "license_type", "tenure_months", "weekly_meetings", "docs_per_week"]
present = [c for c in base_cols if c in df.columns]

model_df = df.dropna(subset=["_y"]).copy()

# Build formula dynamically based on available columns
terms = ["_grp"]
for c in present:
    if model_df[c].dtype == object:
        terms.append(f"C({c})")
    else:
        terms.append(c)

formula = "_y ~ " + " + ".join(terms)

with st.expander("Model formula"):
    st.code(formula)

try:
    model = smf.ols(formula, data=model_df).fit()
    st.subheader("Model summary (key coefficients)")
    st.dataframe(
        model.summary2().tables[1].reset_index().rename(columns={"index": "term"}),
        use_container_width=True,
    )

    st.markdown(
        """
**How to read this:**
- `_grp` coefficient ≈ average change in productivity for adopters vs non-adopters,
  holding other factors constant.
- Categorical terms show differences relative to a reference category.
- Use p-values and confidence intervals for uncertainty.
"""
    )
except Exception as e:
    st.error(f"Regression failed: {e}")

st.success("Inference page loaded successfully.")
