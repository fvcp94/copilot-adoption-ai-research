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
# Synthetic data generator (Cloud-safe)
# ============================================================
def _generate_synthetic_survey(n: int = 1500, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)

    roles = np.array(["IC", "Manager", "Director", "Exec"])
    regions = np.array(["NA", "EMEA", "APAC", "LATAM"])
    devices = np.array(["Windows", "Mac", "Mixed"])
    license_types = np.array(["M365", "M365+Copilot", "Enterprise", "Education"])

    role = rng.choice(roles, size=n, p=[0.68, 0.22, 0.08, 0.02])
    region = rng.choice(regions, size=n, p=[0.45, 0.25, 0.20, 0.10])
    device = rng.choice(devices, size=n, p=[0.62, 0.25, 0.13])
    license_type = rng.choice(license_types, size=n, p=[0.40, 0.30, 0.25, 0.05])

    tenure_months = rng.integers(1, 121, size=n)  # 1..120 months
    weekly_meetings = np.clip(rng.normal(8, 4, size=n).round(), 0, 40).astype(int)
    docs_per_week = np.clip(rng.normal(12, 7, size=n).round(), 0, 60).astype(int)

    # Adoption probability depends on license, role, and workload
    base = 0.18
    base += np.where(license_type == "M365+Copilot", 0.35, 0.0)
    base += np.where(license_type == "Enterprise", 0.20, 0.0)
    base += np.where(role == "Manager", 0.05, 0.0)
    base += np.where(role == "Director", 0.08, 0.0)
    base += np.where(role == "Exec", 0.10, 0.0)
    base += np.clip((weekly_meetings - 8) / 40, -0.10, 0.12)
    base += np.clip((docs_per_week - 12) / 60, -0.10, 0.12)

    p_adopt = np.clip(base, 0.02, 0.92)
    uses_copilot = rng.binomial(1, p_adopt, size=n).astype(bool)

    # Satisfaction + productivity only meaningful for adopters
    # (create realistic but synthetic signals)
    sat_mean = np.where(uses_copilot, 3.8, 3.1)
    sat_mean += np.where(device == "Windows", 0.12, 0.0)
    sat_mean += np.where(region == "NA", 0.08, 0.0)
    satisfaction = np.clip(rng.normal(sat_mean, 0.8, size=n), 1, 5)

    prod_mean = np.where(uses_copilot, 0.22, 0.05)  # 22% vs 5%
    prod_mean += np.clip((docs_per_week - 12) / 200, -0.05, 0.08)
    productivity_gain = np.clip(rng.normal(prod_mean, 0.10, size=n), -0.15, 0.60)

    # Attention check pass: mostly pass, a bit worse for straightliners (simulated)
    straightline_risk = np.clip(rng.normal(0.15, 0.10, size=n), 0, 0.60)
    attention_pass = rng.binomial(1, 1 - straightline_risk, size=n).astype(bool)

    # Open-ended feedback (short synthetic text)
    themes = np.array([
        "saves time", "improves writing", "helps summarize", "needs better accuracy",
        "privacy concerns", "hard to learn", "great for emails", "useful in meetings"
    ])
    feedback = rng.choice(themes, size=n)
    feedback = np.where(
        uses_copilot,
        "Copilot " + feedback.astype(str) + ".",
        "Not using Copilot yet. " + feedback.astype(str) + "."
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
            "column": [
                "respondent_id",
                "role",
                "region",
                "device",
                "license_type",
                "tenure_months",
                "weekly_meetings",
                "docs_per_week",
                "uses_copilot",
                "satisfaction_1to5",
                "productivity_gain_pct",
                "attention_check_pass",
                "open_ended_feedback",
            ],
            "description": [
                "Unique respondent identifier",
                "Job seniority/role category",
                "Geographic region",
                "Primary device ecosystem",
                "Subscription/license type (synthetic)",
                "Tenure at company (months)",
                "Meetings per week (self-reported, synthetic)",
                "Docs created/edited per week (synthetic)",
                "Whether respondent uses Copilot (synthetic adoption)",
                "Satisfaction rating (1–5)",
                "Estimated productivity gain (%)",
                "Passed attention check (survey data quality proxy)",
                "Open-ended feedback text (synthetic)",
            ],
        }
    )

    return df, codebook


# ============================================================
# Ensure dataset exists (create if missing)
# ============================================================
def _ensure_data_exists() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if DATASET_PATH.exists() and CODEBOOK_PATH.exists():
        return

    df, codebook = _generate_synthetic_survey(n=1500, seed=42)
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
