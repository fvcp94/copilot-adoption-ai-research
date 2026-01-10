from __future__ import annotations

import pandas as pd
import plotly.express as px


def hist(df: pd.DataFrame, col: str, title: str):
    return px.histogram(df, x=col, nbins=40, title=title)


def bar_rate(df: pd.DataFrame, group_col: str, rate_col: str, title: str):
    agg = df.groupby(group_col)[rate_col].mean().reset_index()
    return px.bar(agg, x=group_col, y=rate_col, title=title)
