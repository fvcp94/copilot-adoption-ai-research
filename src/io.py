from __future__ import annotations

from pathlib import Path
import pandas as pd
import subprocess
import sys

# ============================================================
# Paths
# ============================================================
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DATASET_PATH = DATA_DIR / "synthetic_survey_data.csv"
CODEBOOK_PATH = DATA_DIR / "codebook.csv"


# ============================================================
# Ensure data exists (auto-generate if missing)
# ============================================================
def _ensure_data_exists() -> None:
    """
    Ensure the synthetic dataset exists.
    If missing (common on Streamlit Cloud), auto-generate it
    by running the data generation script.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # If dataset already exists, nothing to do
    if DATASET_PATH.exists():
        return

    # Path to generator script
    generator_script = ROOT / "data_generation" / "data_generation_process.py"

    if not generator_script.exists():
        raise FileNotFoundError(
            f"Data generator script not found at: {generator_script}"
        )

    # Run the generator script using the current Python executable
    subprocess.check_call(
        [sys.executable, str(generator_script)],
        cwd=str(ROOT),
    )


# ============================================================
# Public loaders
# ============================================================
def load_data() -> pd.DataFrame:
    _ensure_data_exists()
    return pd.read_csv(DATASET_PATH)


def load_codebook() -> pd.DataFrame:
    _ensure_data_exists()

    if CODEBOOK_PATH.exists():
        return pd.read_csv(CODEBOOK_PATH)

    # Graceful fallback if codebook not generated
    return pd.DataFrame(
        {
            "column": [],
            "description": [],
        }
    )
