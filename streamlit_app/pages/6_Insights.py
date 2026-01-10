from __future__ import annotations

import streamlit as st

from src.io import load_data
from src.metrics import adoption_rate

st.title("Insights & Recommendations")
st.caption("Executive-friendly summary: what we found, what to do, and caveats.")

@st.cache_data
def get_df():
    return load_data()

df = get_df()
uw, ww = adoption_rate(df)

st.subheader("Key findings (synthetic demo)")
st.markdown(
f"""
- Adoption rate (unweighted): **{uw:.3f}**
- Adoption rate (weighted): **{ww:.3f}** (post-stratification demo)
- Adoption correlates with higher productivity (see causal page for PSM estimate).
"""
)

st.subheader("Recommended actions")
st.markdown(
"""
1. **Target onboarding** to segments with high M365 usage but lower adoption.
2. **Reduce trust concerns** via clearer controls, transparency messaging, and security UX.
3. Improve perceived **usefulness** via role-based workflows (Sales vs Finance vs Engineering).
"""
)

st.subheader("Caveats")
st.markdown(
"""
- Data is synthetic for demonstration.
- Self-reported productivity is biased; telemetry is better when available.
- PSM helps but still requires careful causal identification.
"""
)
