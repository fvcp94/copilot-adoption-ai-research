from __future__ import annotations

import streamlit as st
import pandas as pd

from llm_agents.survey_question_agent import run_survey_question_agent
from llm_agents.openrouter_client import OpenRouterClient

st.title("🤖 AI-Enhanced: Survey Question Generator & Evaluator")
st.caption("Uses OpenRouter (free models supported via :free). Falls back to deterministic mode if rate-limited or not configured.")

or_client = OpenRouterClient()
st.info(
    f"OpenRouter configured: **{or_client.is_configured()}** | Model: **{or_client.ensure_free_model(st.secrets.get('OPENROUTER_MODEL', '') if hasattr(st, 'secrets') else '') or 'mistral/devstral-2-2512:free'}**"
)

objective = st.text_area(
    "Research objective",
    value="I want to measure user satisfaction and adoption barriers for an AI assistant at work.",
    height=120
)

if st.button("Generate & Evaluate"):
    result = run_survey_question_agent(objective)

    st.subheader("Constructs identified")
    st.write(result.constructs)

    st.subheader("Generated survey questions")
    gen_df = pd.DataFrame([{
        "Construct": g.construct,
        "Question": g.question,
        "Response scale": g.response_scale,
        "Rationale": g.rationale
    } for g in result.generated])
    st.dataframe(gen_df, use_container_width=True)

    st.subheader("Quality evaluation")
    eval_df = pd.DataFrame([{
        "Question": e["question"],
        "Flesch reading ease": e["flesch_reading_ease"],
        "Issue count": e["issue_count"],
        "Issues": ", ".join([i["type"] for i in e["issues"]]) if e["issues"] else ""
    } for e in result.evaluations])
    st.dataframe(eval_df, use_container_width=True)

    if result.recommended_rewrites:
        st.subheader("Recommended rewrites")
        for r in result.recommended_rewrites:
            with st.expander(r["original"]):
                st.write("**Suggested rewrite:**", r["suggested"])
                st.write("**Issues detected:**")
                st.json(r["issues"])
    else:
        st.success("No major issues detected by the deterministic checks.")

    st.subheader("Evidence-based guidance (retrieved)")
    if result.rag_guidance:
        for hit in result.rag_guidance:
            with st.expander(f"{hit['source']} (score={hit['score']})"):
                st.code(hit["snippet"])
    else:
        st.info("No KB guidance found. Add docs to docs/survey_methodology_kb/*.md")
