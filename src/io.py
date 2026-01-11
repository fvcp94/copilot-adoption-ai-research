from __future__ import annotations

from pathlib import Path
import pandas as pd

# Project root (repo root)
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

DATASET_PATH = DATA_DIR / "synthetic_survey_data.csv"
CODEBOOK_PATH = DATA_DIR / "codebook.csv"


def _ensure_data_exists() -> None:
    """
    Ensure synthetic dataset + codebook exist.
    If missing (common on Streamlit Cloud), auto-generate them.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if DATASET_PATH.exists() and CODEBOOK_PATH.exists():
        return

    # Try generating data using the generator script/module
    # This assumes your generator lives under data_generation/
    try:
        from data_generation.generate_survey_data import generate_synthetic_survey_data  # type: ignore
    except Exception:
        # Fallback: try an alternate common name
        from data_generation.data_generation_process import generate_synthetic_survey_data  # type: ignore

    df, codebook = generate_synthetic_survey_data(
        n=1500,
        seed=42,
    )

    df.to_csv(DATASET_PATH, index=False)
    codebook.to_csv(CODEBOOK_PATH, index=False)


def load_data() -> pd.DataFrame:
    _ensure_data_exists()
    return pd.read_csv(DATASET_PATH)


def load_codebook() -> pd.DataFrame:
    _ensure_data_exists()
    return pd.read_csv(CODEBOOK_PATH)
