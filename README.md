# Copilot Adoption — AI-Powered Survey Research Intelligence Platform (Demo)

This repo is an end-to-end **survey research + analytics** demo (synthetic data) with an **AI-enhancement layer** powered by **OpenRouter** (free models supported via `:free`).

## Features

### Traditional research foundation
- Synthetic survey dataset generator (with sampling bias + raked weights)
- Data quality checks (missingness, attention check, straight-lining, dropout)
- Inferential analysis (t-test, logistic regression, OLS regression)
- Experimental analysis (onboarding treatment)
- Causal demo (propensity score matching)

### AI enhancement layer
- Survey Question Agent:
  - Extracts constructs from a research objective (LLM)
  - Generates survey questions (LLM)
  - Evaluates for bias/clarity/readability + provides deterministic rewrites
  - Retrieves methodology guidance from a local KB (RAG-lite)

## Local run (recommended)
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt

# Optional (for LLM calls):
# Create .env (do NOT commit) with:
# OPENROUTER_API_KEY=...
# OPENROUTER_MODEL=mistral/devstral-2-2512:free
# APP_NAME=Copilot Survey Research Demo
# APP_URL=http://localhost:8501

# Generate data (optional — app can auto-generate):
python data_generation/data_generation_process.py

# Run Streamlit:
streamlit run streamlit_app/app.py
```

## Deploy (Streamlit Community Cloud)
1. Push this repo to GitHub.
2. Create a new Streamlit app:
   - Main file path: `streamlit_app/app.py`
3. Add Streamlit **Secrets** (Settings → Secrets):
```toml
OPENROUTER_API_KEY="YOUR_KEY"
OPENROUTER_MODEL="mistral/devstral-2-2512:free"
APP_NAME="Copilot Survey Research Demo"
APP_URL="https://YOUR-APP.streamlit.app"
```
4. Deploy. The app auto-generates synthetic data if `data/synthetic_survey_data.csv` is missing.

## OpenRouter “Free” usage
Use models that end with `:free` to avoid spending credits. Free models have daily request caps; if the cap is hit, the app falls back to deterministic question templates.

---

**Disclaimer:** This is synthetic demo data and not Microsoft product telemetry.
