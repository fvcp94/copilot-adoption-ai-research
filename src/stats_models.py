from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf


def cohens_d(x: pd.Series, y: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce").dropna()
    y = pd.to_numeric(y, errors="coerce").dropna()
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    nx, ny = len(x), len(y)
    vx, vy = x.var(ddof=1), y.var(ddof=1)
    pooled = ((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2)
    return float((x.mean() - y.mean()) / np.sqrt(pooled))


def two_sample_ttest(df: pd.DataFrame, outcome: str, group: str = "copilot_adopted"):
    a = pd.to_numeric(df.loc[df[group] == 1, outcome], errors="coerce").dropna()
    b = pd.to_numeric(df.loc[df[group] == 0, outcome], errors="coerce").dropna()

    res = stats.ttest_ind(a, b, equal_var=False, nan_policy="omit")
    d = cohens_d(a, b)
    return {
        "mean_group1": float(a.mean()) if len(a) else float("nan"),
        "mean_group0": float(b.mean()) if len(b) else float("nan"),
        "diff": float(a.mean() - b.mean()) if len(a) and len(b) else float("nan"),
        "t_stat": float(res.statistic),
        "p_value": float(res.pvalue),
        "cohens_d": d,
        "n1": int(len(a)),
        "n0": int(len(b)),
    }


def adoption_logit(df: pd.DataFrame):
    formula = (
        "copilot_adopted ~ m365_usage + ai_openness + C(company_size) + C(job_family) + C(region) + C(tenure_bucket) + onboarding_treatment"
    )
    return smf.logit(formula=formula, data=df).fit(disp=False)


def productivity_ols(df: pd.DataFrame):
    formula = (
        "productivity_score ~ copilot_adopted + m365_usage + ai_openness + C(company_size) + C(job_family) + C(region) + C(tenure_bucket)"
    )
    return smf.ols(formula=formula, data=df).fit()
