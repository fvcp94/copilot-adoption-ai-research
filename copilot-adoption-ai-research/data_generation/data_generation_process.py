from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd


@dataclass
class Config:
    n: int = 5000
    seed: int = 42
    out_dir: str = "data"
    oversample_enterprise: float = 1.35
    oversample_engineering: float = 1.30
    oversample_heavy_m365: float = 1.25
    base_adoption_rate: float = 0.35
    treatment_effect_logit: float = 0.18
    noise_sd: float = 0.65
    p_dropout: float = 0.03
    p_straightline: float = 0.04
    p_failed_attention: float = 0.03


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def clip01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0, 1)


def likert_from_latent(latent: np.ndarray, k: int = 5) -> np.ndarray:
    qs = np.quantile(latent, np.linspace(0, 1, k + 1))
    qs = np.unique(qs)
    if len(qs) < k + 1:
        qs = np.linspace(np.min(latent), np.max(latent), k + 1)
    out = np.digitize(latent, qs[1:-1], right=True) + 1
    return out.astype(int)


def generate_target_population_margins() -> Dict[str, Dict[str, float]]:
    return {
        "company_size": {"SMB": 0.45, "Mid": 0.30, "Enterprise": 0.25},
        "job_family": {"Engineering": 0.22, "Sales": 0.18, "Ops": 0.20, "HR": 0.12, "Finance": 0.14, "Other": 0.14},
        "region": {"NA": 0.42, "EU": 0.28, "APAC": 0.22, "LATAM": 0.08},
        "tenure_bucket": {"0-1": 0.20, "1-3": 0.30, "3-7": 0.30, "7+": 0.20},
    }


def weighted_rake(df: pd.DataFrame, margins: Dict[str, Dict[str, float]], max_iter: int = 50, tol: float = 1e-6):
    w = np.ones(len(df), dtype=float)
    for _ in range(max_iter):
        w_prev = w.copy()
        for col, target in margins.items():
            cur = df.groupby(col).apply(lambda g: w[g.index].sum()).to_dict()
            total = w.sum()
            for cat, target_prop in target.items():
                cur_mass = cur.get(cat, 0.0)
                desired_mass = target_prop * total
                if cur_mass > 0:
                    adj = desired_mass / cur_mass
                    mask = (df[col] == cat).to_numpy()
                    w[mask] *= adj
        if np.max(np.abs(w - w_prev)) < tol:
            break
    return w / np.mean(w)


def build_codebook() -> Dict[str, str]:
    return {
        "respondent_id": "Unique respondent identifier",
        "company_size": "SMB / Mid / Enterprise",
        "job_family": "Engineering / Sales / Ops / HR / Finance / Other",
        "region": "NA / EU / APAC / LATAM",
        "tenure_bucket": "0-1 / 1-3 / 3-7 / 7+ years",
        "m365_usage": "Microsoft 365 usage intensity (0-1)",
        "ai_openness": "Openness to AI tools (0-1)",
        "eligible_onboarding": "Eligibility for onboarding experiment (1/0)",
        "onboarding_treatment": "Onboarding treatment assignment (1/0)",
        "copilot_adopted": "Copilot adoption indicator (1/0)",
        "copilot_usage_freq": "0=never, 1=monthly, 2=weekly, 3=daily",
        "productivity_score": "Self-reported productivity score (0-100)",
        "trust_concern": "Trust/privacy concerns (0-1, higher=worse)",
        "ease_of_use": "Ease of use (0-1, adopters only)",
        "usefulness": "Perceived usefulness (0-1, adopters only)",
        "reliability": "Perceived reliability (0-1, adopters only)",
        "sat_overall": "Overall satisfaction (Likert 1-5, adopters only)",
        "sat_helpfulness": "Helpfulness satisfaction (Likert 1-5, adopters only)",
        "sat_time_saved": "Time saved satisfaction (Likert 1-5, adopters only)",
        "sat_quality": "Output quality satisfaction (Likert 1-5, adopters only)",
        "failed_attention_check": "Failed attention check (1/0)",
        "straight_lining_flag": "Straight-lining detected (1/0)",
        "dropout_flag": "Partial completion/dropout (1/0)",
        "weight_raked": "Post-stratification (raked) weight (mean=1)"
    }


def simulate_sample(cfg: Config) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed)

    company_size = rng.choice(["SMB", "Mid", "Enterprise"], p=[0.45, 0.30, 0.25], size=cfg.n)
    job_family = rng.choice(["Engineering", "Sales", "Ops", "HR", "Finance", "Other"],
                            p=[0.22, 0.18, 0.20, 0.12, 0.14, 0.14], size=cfg.n)
    region = rng.choice(["NA", "EU", "APAC", "LATAM"], p=[0.42, 0.28, 0.22, 0.08], size=cfg.n)
    tenure_bucket = rng.choice(["0-1", "1-3", "3-7", "7+"], p=[0.20, 0.30, 0.30, 0.20], size=cfg.n)

    base_usage = rng.beta(2.2, 2.0, size=cfg.n)
    base_usage += (company_size == "Enterprise") * 0.08 + (job_family == "Engineering") * 0.06
    m365_usage = clip01(base_usage)

    ai_openness = clip01(rng.normal(0.55, 0.18, size=cfg.n) + (job_family == "Engineering") * 0.08)

    resp_score = np.ones(cfg.n, dtype=float)
    resp_score *= np.where(company_size == "Enterprise", cfg.oversample_enterprise, 1.0)
    resp_score *= np.where(job_family == "Engineering", cfg.oversample_engineering, 1.0)
    resp_score *= (1 + cfg.oversample_heavy_m365 * (m365_usage - 0.5))
    resp_prob = resp_score / resp_score.max()
    is_in_sample = rng.uniform(0, 1, size=cfg.n) < resp_prob

    df = pd.DataFrame({
        "company_size": company_size,
        "job_family": job_family,
        "region": region,
        "tenure_bucket": tenure_bucket,
        "m365_usage": m365_usage,
        "ai_openness": ai_openness,
        "in_sample": is_in_sample
    })
    df = df[df["in_sample"]].drop(columns=["in_sample"]).reset_index(drop=True)

    eligible = (df["m365_usage"] > 0.35).astype(int)
    treatment = np.where(eligible == 1, rng.integers(0, 2, size=len(df)), 0)
    df["eligible_onboarding"] = eligible
    df["onboarding_treatment"] = treatment

    logit = (
        math.log(cfg.base_adoption_rate / (1 - cfg.base_adoption_rate))
        + 1.25 * (df["m365_usage"].to_numpy() - 0.5)
        + 1.05 * (df["ai_openness"].to_numpy() - 0.5)
        + 0.35 * (df["company_size"].eq("Enterprise").to_numpy())
        + 0.25 * (df["job_family"].eq("Engineering").to_numpy())
        + cfg.treatment_effect_logit * df["onboarding_treatment"].to_numpy()
        + rng.normal(0, cfg.noise_sd, size=len(df))
    )
    p_adopt = sigmoid(logit)
    adopted = rng.uniform(0, 1, size=len(df)) < p_adopt
    df["copilot_adopted"] = adopted.astype(int)

    freq_latent = (
        0.8 * (df["m365_usage"].to_numpy())
        + 0.6 * (df["ai_openness"].to_numpy())
        + 0.3 * (df["job_family"].eq("Engineering").to_numpy().astype(float))
        + rng.normal(0, 0.6, size=len(df))
    )
    freq = np.zeros(len(df), dtype=int)
    bins = np.quantile(freq_latent[df["copilot_adopted"] == 1], [0.25, 0.55, 0.80]) if df["copilot_adopted"].sum() > 10 else [0.0, 0.3, 0.6]
    idx = df["copilot_adopted"].to_numpy() == 1
    freq[idx] = 1 + np.digitize(freq_latent[idx], bins, right=True)
    df["copilot_usage_freq"] = freq

    base_prod = 50 + 18 * (df["m365_usage"] - 0.5) + 12 * (df["ai_openness"] - 0.5)
    adopt_lift = 6 + 8 * (df["copilot_usage_freq"] / 3.0)
    prod = base_prod + df["copilot_adopted"] * adopt_lift + rng.normal(0, 10, size=len(df))
    df["productivity_score"] = np.clip(prod, 0, 100).round(1)

    trust_concern = clip01(rng.beta(2.0, 3.0, size=len(df)) - 0.15 * df["copilot_adopted"] + rng.normal(0, 0.08, size=len(df)))
    ease = clip01(0.55 + 0.25 * df["ai_openness"] + 0.15 * df["m365_usage"] + 0.10 * df["copilot_adopted"] + rng.normal(0, 0.12, size=len(df)))
    usefulness = clip01(0.40 + 0.35 * (df["copilot_usage_freq"] / 3.0) + 0.15 * df["m365_usage"] + rng.normal(0, 0.12, size=len(df)))
    reliability = clip01(0.50 + 0.20 * df["copilot_adopted"] + rng.normal(0, 0.15, size=len(df)))

    latent_sat = (0.35 * usefulness + 0.25 * ease + 0.25 * reliability - 0.30 * trust_concern + rng.normal(0, 0.18, size=len(df)))

    df["trust_concern"] = trust_concern.round(3)
    df["ease_of_use"] = np.where(df["copilot_adopted"] == 1, ease.round(3), np.nan)
    df["usefulness"] = np.where(df["copilot_adopted"] == 1, usefulness.round(3), np.nan)
    df["reliability"] = np.where(df["copilot_adopted"] == 1, reliability.round(3), np.nan)

    for item in ["sat_overall", "sat_helpfulness", "sat_time_saved", "sat_quality"]:
        item_latent = latent_sat + rng.normal(0, 0.25, size=len(df))
        lik = likert_from_latent(item_latent, k=5)
        df[item] = np.where(df["copilot_adopted"] == 1, lik, np.nan)

    failed_attention = rng.uniform(0, 1, size=len(df)) < cfg.p_failed_attention
    straightline = rng.uniform(0, 1, size=len(df)) < cfg.p_straightline
    dropout = rng.uniform(0, 1, size=len(df)) < cfg.p_dropout

    df["failed_attention_check"] = failed_attention.astype(int)
    df["straight_lining_flag"] = straightline.astype(int)
    df["dropout_flag"] = dropout.astype(int)

    for col in ["sat_time_saved", "sat_quality"]:
        df.loc[df["dropout_flag"] == 1, col] = np.nan

    sl_mask = df["straight_lining_flag"] == 1
    if sl_mask.any():
        constant = rng.integers(1, 6, size=sl_mask.sum())
        for col in ["sat_overall", "sat_helpfulness", "sat_time_saved", "sat_quality"]:
            df.loc[sl_mask & (df["copilot_adopted"] == 1), col] = constant

    df.insert(0, "respondent_id", np.arange(1, len(df) + 1))
    return df


def main():
    cfg = Config()
    os.makedirs(cfg.out_dir, exist_ok=True)

    df = simulate_sample(cfg)
    margins = generate_target_population_margins()
    df["weight_raked"] = weighted_rake(df, margins)

    pop_margins = []
    for col, dist in margins.items():
        for k, v in dist.items():
            pop_margins.append({"dimension": col, "category": k, "target_prop": v})
    pd.DataFrame(pop_margins).to_csv(os.path.join(cfg.out_dir, "target_population_margins.csv"), index=False)

    with open(os.path.join(cfg.out_dir, "codebook.json"), "w", encoding="utf-8") as f:
        json.dump(build_codebook(), f, indent=2)

    out_path = os.path.join(cfg.out_dir, "synthetic_survey_data.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path} (rows={len(df)})")


if __name__ == "__main__":
    main()
