from __future__ import annotations

import numpy as np
import pandas as pd


def weighted_mean(x: pd.Series, w: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce")
    w = pd.to_numeric(w, errors="coerce")
    mask = x.notna() & w.notna()
    if mask.sum() == 0:
        return float("nan")
    return float(np.average(x[mask].to_numpy(), weights=w[mask].to_numpy()))


def adoption_rate(df: pd.DataFrame, weight_col: str = "weight_raked") -> tuple[float, float]:
    unweighted = float(df["copilot_adopted"].mean())
    weighted = weighted_mean(df["copilot_adopted"], df[weight_col]) if weight_col in df.columns else float("nan")
    return unweighted, weighted


def completion_rate(df: pd.DataFrame) -> float:
    if "dropout_flag" not in df.columns:
        return float("nan")
    return float((df["dropout_flag"] == 0).mean())


def attention_pass_rate(df: pd.DataFrame) -> float:
    if "failed_attention_check" not in df.columns:
        return float("nan")
    return float((df["failed_attention_check"] == 0).mean())
