"""Loading + forecasting helpers for the dashboard."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import pipeline as P  # noqa: E402  (shared with the notebook)

MODEL_KEYS = ["baseline", "rf", "xgb", "stack"]
MAX_HORIZON = 36            # months beyond the last month of data
ALL = "All Sri Lanka"


def load_metrics() -> dict:
    return json.loads((ROOT / "artifacts" / "metrics.json").read_text(encoding="utf-8"))


def load_panel() -> pd.DataFrame:
    return pd.read_csv(ROOT / "data" / "processed" / "monthly_panel.csv", parse_dates=["month_start"])


def load_onestep() -> pd.DataFrame:
    return pd.read_csv(ROOT / "artifacts" / "predictions_onestep.csv", parse_dates=["month_start"])


def load_backtest() -> pd.DataFrame:
    return pd.read_csv(ROOT / "artifacts" / "backtest_recursive.csv", parse_dates=["month_start"])


def load_models() -> dict:
    return {k: joblib.load(ROOT / "models" / f"deploy_{k}.joblib") for k in MODEL_KEYS}


def forecast_all(models: dict, panel: pd.DataFrame, n: int = MAX_HORIZON) -> pd.DataFrame:
    """Recursive forecast of every district, for every model, `n` months ahead."""
    clim = P.make_climatology(panel)
    parts = []
    for k, m in models.items():
        f = P.recursive_forecast(m, panel, n, clim)
        f["model"] = k
        parts.append(f)
    return pd.concat(parts, ignore_index=True)


def region_series(fc: pd.DataFrame, model: str, region: str, sigma: dict) -> pd.DataFrame:
    """Forecast path for one model and region, with an approximate 80% range."""
    f = fc[fc["model"] == model]
    if region == ALL:
        g = f.groupby(["month_start", "horizon"], as_index=False)["pred"].sum()
        tbl = sigma[model]["national"]
    else:
        g = f[f["district"] == region][["month_start", "horizon", "pred"]].copy()
        tbl = sigma[model]["district"]
    lo, hi = [], []
    for p, h in zip(g["pred"], g["horizon"]):
        a, b = P.band(p, h, tbl)
        lo.append(float(a)); hi.append(float(b))
    g["low"], g["high"] = lo, hi
    return g.sort_values("month_start").reset_index(drop=True)


def region_actual(panel: pd.DataFrame, region: str) -> pd.Series:
    d = panel if region == ALL else panel[panel["district"] == region]
    return d.groupby("month_start")["cases"].sum()
