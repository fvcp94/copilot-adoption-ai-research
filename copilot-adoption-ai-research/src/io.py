from __future__ import annotations

import json
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"


def load_data() -> pd.DataFrame:
    path = DATA_DIR / "synthetic_survey_data.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing dataset at {path}. The app can auto-generate it from Streamlit, or run data_generation script."
        )
    return pd.read_csv(path)


def load_codebook() -> dict:
    path = DATA_DIR / "codebook.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_population_margins() -> pd.DataFrame:
    path = DATA_DIR / "target_population_margins.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)
