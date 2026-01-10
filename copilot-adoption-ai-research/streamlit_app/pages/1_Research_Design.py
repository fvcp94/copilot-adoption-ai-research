from __future__ import annotations

import streamlit as st
from src.io import load_population_margins

st.title("Research Design")
st.caption("Survey methodology, sampling strategy, and weighting transparency (synthetic demo).")

st.markdown(
"""
### Study framing
**Goal:** Understand adoption patterns and satisfaction drivers, and estimate productivity impact.

**Key hypotheses**
1. Adoption is higher among users with higher Microsoft 365 usage intensity and AI openness.
2. Adoption is associated with higher self-reported productivity (confounded → causal methods needed).
3. Satisfaction is driven by usefulness, ease-of-use, reliability, and trust concerns.

### Sampling strategy (demo)
Dataset includes intentional **sample bias** to demonstrate:
- post-stratification / raking weights
- difference between unweighted vs weighted estimates
"""
)

margins = load_population_margins()
with st.expander("Target population margins used for weighting"):
    if len(margins):
        st.dataframe(margins, use_container_width=True)
    else:
        st.info("target_population_margins.csv not found (will be generated automatically).")
