from __future__ import annotations

import pandas as pd


def _col(df: pd.DataFrame, *candidates: str) -> str:
    """Return the first existing column name from candidates."""
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"None of these columns exist in dataframe: {candidates}. Found: {list(df.columns)}")


def adoption_rate(df: pd.DataFrame) -> float:
    """
    Adoption rate of Copilot.
    Supports both legacy column name `copilot_adopted` and newer `uses_copilot`.
    """
    c = _col(df, "copilot_adopted", "uses_copilot")
    return float(pd.to_numeric(df[c], errors="coerce").fillna(0).mean())


def attention_pass_rate(df: pd.DataFrame) -> float:
    """
    Rate of passing attention checks.
    Supports both legacy `attention_check_passed` and `attention_check_pass`.
    """
    c = _col(df, "attention_check_passed", "attention_check_pass")
    return float(pd.to_numeric(df[c], errors="coerce").fillna(0).mean())
