from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

# ============================================================
# Paths
# ============================================================
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DATASET_PATH = DATA_DIR / "synthetic_survey_data.csv"
CODEBOOK_PATH = DATA_DIR / "codebook.csv"


# ============================================================
# Synthetic survey data generator (Cloud-safe)
# ============================================================
def _generate_synthetic_survey(
    n: int = 1500,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)

    roles = np.array(["IC", "Manager", "Director", "Exec"])
    regions = np.array(["NA", "EMEA", "APAC", "LATAM"])
    devices = np.array(["Windows", "Mac", "Mixed"])
    license_types = np.array(["M365", "M365+Copilot", "Enterprise", "Education"])

    role = rng.choice(roles, size=n, p=[0.68, 0.22, 0.08, 0.02])
    region = rng.choice(regions, size=n, p=[0.45, 0.25, 0.20, 0.10])
    device = rng.choice(devices, size=n, p=[0.62, 0.25, 0.13])
    license_type = rng.choice(license_types, size=n, p=[0.40, 0.30, 0.25, 0.05])

    tenure_months = rng.integers(1, 121, size=n)
    weekly_meetings = np.clip(rng.normal(8, 4, size=n).round(), 0, 40).astype(int)
    docs_per_week = np.clip(rng.normal(12, 7, size=n).round(), 0, 60).astype(int)

    base = 0.18
    base += np.where(license_type == "M365+Copilot", 0.35, 0.0)
    base += np.where(license_type == "Enterprise", 0.20, 0.0)
    base += np.where(role == "Manager", 0.05, 0.0)
    base += np.where(role == "Director", 0.08, 0.0)
    base += np.where(role == "Exec", 0.10, 0.0)

    p_adopt = np.clip(base, 0.02, 0.95)
    uses_copilot = rng.binomial(1, p_adopt, size=n).astype(bool)

    satisfaction = np.clip(
        rng.normal(np.where(uses_copilot, 3.8, 3.1), 0.8, size=n),
        1,
        5,
    )

    productivity_gain = np.clip(
        rng.normal(np.where(uses_copilot, 0.22, 0.05), 0.10, size=n),
        -0.15,
        0.60,
    )

    attention_pass = rng.binomial(1, 0.83, size=n).astype(bool)

    feedback_bank = np.array(
        [
            "saves time",
            "improves writing",
            "helps summarize",
            "needs better accuracy",
            "privacy concerns",
            "hard to learn",
            "great for emails",
            "useful in meetings",
        ]
    )

    feedback = rng.choice(feedback_bank, size=n)
    feedback = np.where(
        uses_copilot,
        "Copilot " + feedback + ".",
        "Not using Copilot yet. " + feedback + ".",
    )

    df = pd.DataFrame(
        {
            "respondent_id": np.arange(1, n + 1),
            "role": role,
            "region": region,
            "device": device,
            "license_type": license_type,
            "tenure_months": tenure_months,
            "weekly_meetings": weekly_meetings,
            "docs_per_week": docs_per_week,
            "uses_copilot": uses_copilot,
            "satisfaction_1to5": satisfaction.round(2),
            "productivity_gain_pct": (productivity_gain * 100).round(2),
            "attention_check_pass": attention_pass,
            "open_ended_feedback": feedback,
        }
    )

    codebook = pd.DataFrame(
        {
            "column": df.columns,
            "description": [
                "Unique respondent identifier",
                "Job role / seniority",
                "Geographic region",
                "Primary device ecosystem",
                "License / subscription type",
                "Tenure in months",
                "Meetings per week",
                "Documents handled per week",
                "Whether respondent uses Copilot",
                "Satisfaction rating (1–5)",
                "Estimated productivity gain (%)",
                "Passed attention check",
                "Open-ended qualitative feedback",
            ],
        }
    )

    return df, codebook


# ============================================================
# Ensure dataset exists
# ============================================================
def _ensure_data_exists() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if DATASET_PATH.exists() and CODEBOOK_PATH.exists():
        return

    df, codebook = _generate_synthetic_survey()
    df.to_csv(DATASET_PATH, index=False)
    codebook.to_csv(CODEBOOK_PATH, index=False)


# ============================================================
# Public loaders
# ============================================================
def load_data() -> pd.DataFrame:
    _ensure_data_exists()
    return pd.read_csv(DATASET_PATH)


def load_codebook() -> pd.DataFrame:
    _ensure_data_exists()
    return pd.read_csv(CODEBOOK_PATH)


def load_population_margins() -> dict:
    """
    Synthetic population margins used for weighting demonstrations.
    """
    return {
        "role": {
            "IC": 0.68,
            "Manager": 0.22,
            "Director": 0.08,
            "Exec": 0.02,
        },
        "region": {
            "NA": 0.45,
            "EMEA": 0.25,
            "APAC": 0.20,
            "LATAM": 0.10,
        },
        "device": {
            "Windows": 0.62,
            "Mac": 0.25,
            "Mixed": 0.13,
        },
        "license_type": {
            "M365": 0.40,
            "M365+Copilot": 0.30,
            "Enterprise": 0.25,
            "Education": 0.05,
        },
    }
