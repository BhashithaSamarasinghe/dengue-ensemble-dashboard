"""Shared code for the dengue ensemble project.

Used by BOTH the notebook (training) and the Streamlit app (forecasting), so the
features at prediction time are built by exactly the same code as at training time.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --------------------------------------------------------------------------
# 1. Data -> monthly district panel
# --------------------------------------------------------------------------
BASE_COLS = ["district", "month_start", "cases", "rain", "tmax", "tmin", "hum",
             "population", "lat", "lon", "coastal"]


def load_raw(path):
    return pd.read_csv(path, parse_dates=["start.date", "end.date"])


def build_monthly_panel(df: pd.DataFrame, min_weeks: int = 3) -> pd.DataFrame:
    """Weekly district rows -> one row per district per month.

    A month holds 4 or 5 weeks, so a plain sum would make 5-week months look
    busier. We therefore take the weekly MEAN and scale it to the month length:
        monthly_total = weekly_mean * days_in_month / 7
    Months with fewer than `min_weeks` weeks of data (the partial first/last
    month) are dropped.
    """
    d = df.copy()
    d["month_start"] = d["start.date"].dt.to_period("M").dt.to_timestamp()
    panel = (d.groupby(["district", "month_start"])
               .agg(weeks=("cases", "size"),
                    cases_w=("cases", "mean"),
                    rain_w=("rainfall_mm", "mean"),
                    tmax=("temp_max_c", "mean"),
                    tmin=("temp_min_c", "mean"),
                    hum=("humidity_pct", "mean"),
                    population=("population", "mean"),
                    lat=("latitude", "first"),
                    lon=("longitude", "first"),
                    coastal=("coastal_district", "first"))
               .reset_index())
    scale = panel["month_start"].dt.days_in_month / 7.0
    panel["cases"] = panel["cases_w"] * scale
    panel["rain"] = panel["rain_w"] * scale
    panel = panel[panel["weeks"] >= min_weeks].drop(columns=["cases_w", "rain_w", "weeks"])
    return panel[BASE_COLS].sort_values(["district", "month_start"]).reset_index(drop=True)


# --------------------------------------------------------------------------
# 2. Features (only PAST information: every feature is a lag >= 1 month)
# --------------------------------------------------------------------------
FEATURES = [
    "month_sin", "month_cos",
    "lcases_lag1", "lcases_lag2", "lcases_lag3", "lcases_lag6", "lcases_lag12",
    "lcases_roll3", "lcases_roll6",
    "lnat_lag1", "lnat_lag2", "lnat_roll3",
    "lrain_lag1", "lrain_lag2", "lrain_lag3", "lrain_roll3",
    "tmax_lag1", "tmin_lag1", "hum_lag1",
    "district_code", "lat", "lon", "coastal", "lpop",
]


def add_features(panel: pd.DataFrame) -> pd.DataFrame:
    p = panel.sort_values(["district", "month_start"]).reset_index(drop=True).copy()
    districts = sorted(p["district"].unique())
    p["district_code"] = pd.Categorical(p["district"], categories=districts).codes
    p["y"] = np.log1p(p["cases"])                      # target on log scale
    p["lrain"] = np.log1p(p["rain"])
    p["lpop"] = np.log(p["population"])
    p["month"] = p["month_start"].dt.month
    p["month_sin"] = np.sin(2 * np.pi * p["month"] / 12)
    p["month_cos"] = np.cos(2 * np.pi * p["month"] / 12)

    g = p.groupby("district")
    for k in (1, 2, 3, 6, 12):
        p[f"lcases_lag{k}"] = g["y"].shift(k)
    p["lcases_roll3"] = p[[f"lcases_lag{k}" for k in (1, 2, 3)]].mean(axis=1, skipna=False)
    shifted = [g["y"].shift(k) for k in range(1, 7)]
    p["lcases_roll6"] = pd.concat(shifted, axis=1).mean(axis=1, skipna=False)

    # national wave: total cases in the whole country in previous months
    nat = p.groupby("month_start")["cases"].sum(min_count=len(districts)).sort_index()
    lnat = np.log1p(nat)
    nat_f = pd.DataFrame({
        "lnat_lag1": lnat.shift(1), "lnat_lag2": lnat.shift(2),
        "lnat_roll3": pd.concat([lnat.shift(k) for k in (1, 2, 3)], axis=1).mean(axis=1, skipna=False),
    })
    for c in nat_f.columns:
        p[c] = p["month_start"].map(nat_f[c])

    for k in (1, 2, 3):
        p[f"lrain_lag{k}"] = g["lrain"].shift(k)
    p["lrain_roll3"] = p[[f"lrain_lag{k}" for k in (1, 2, 3)]].mean(axis=1, skipna=False)
    p["tmax_lag1"] = g["tmax"].shift(1)
    p["tmin_lag1"] = g["tmin"].shift(1)
    p["hum_lag1"] = g["hum"].shift(1)
    return p


def modelling_rows(feat: pd.DataFrame) -> pd.DataFrame:
    """Rows that have a known target and a complete feature vector."""
    ok = feat["cases"].notna() & feat[FEATURES].notna().all(axis=1)
    return feat[ok].reset_index(drop=True)


# --------------------------------------------------------------------------
# 3. Time-aware validation
# --------------------------------------------------------------------------
def month_block_splits(groups, n_splits=4, val_len=12):
    """Expanding-window splits that respect time (never shuffle).

    `groups` = month ordinal of every row. The last n_splits*val_len months are
    used as consecutive validation windows; each model is trained on everything
    BEFORE its window.
    """
    groups = np.asarray(groups)
    months = np.sort(np.unique(groups))
    first_val = len(months) - n_splits * val_len
    for i in range(n_splits):
        lo = months[first_val + i * val_len]
        hi = months[first_val + (i + 1) * val_len - 1]
        tr = np.where(groups < lo)[0]
        va = np.where((groups >= lo) & (groups <= hi))[0]
        yield tr, va


class TimeStacker(BaseEstimator, RegressorMixin):
    """Stacking ensemble for time-ordered data.

    Level-0 learners are scored with EXPANDING-WINDOW out-of-fold predictions
    (each block of months is predicted by models trained only on earlier months),
    then a Ridge meta-learner (non-negative weights) learns how to blend them.
    sklearn's StackingRegressor cannot do this because it needs a partitioning
    CV, which would let the future leak into the meta-learner's training data.
    """

    def __init__(self, estimators, final_estimator=None, min_train_months=60, block=12):
        self.estimators = estimators
        self.final_estimator = final_estimator
        self.min_train_months = min_train_months
        self.block = block

    def fit(self, X, y, groups):
        X = X.reset_index(drop=True)
        y = np.asarray(y)
        groups = np.asarray(groups)
        months = np.sort(np.unique(groups))
        oof = np.full((len(X), len(self.estimators)), np.nan)
        start = self.min_train_months
        while start < len(months):
            val_months = months[start:start + self.block]
            tr = np.where(groups < val_months[0])[0]
            va = np.where(np.isin(groups, val_months))[0]
            for j, (_, est) in enumerate(self.estimators):
                m = clone(est).fit(X.iloc[tr], y[tr])
                oof[va, j] = m.predict(X.iloc[va])
            start += self.block
        mask = ~np.isnan(oof).any(axis=1)
        self.meta_ = clone(self.final_estimator or Ridge(alpha=1.0, positive=True))
        self.meta_.fit(oof[mask], y[mask])
        self.oof_rows_ = int(mask.sum())
        self.names_ = [n for n, _ in self.estimators]
        self.fitted_ = [clone(est).fit(X, y) for _, est in self.estimators]
        return self

    def base_predictions(self, X):
        return np.column_stack([m.predict(X) for m in self.fitted_])

    def predict(self, X):
        return self.meta_.predict(self.base_predictions(X))

    @property
    def weights_(self):
        return dict(zip(self.names_, self.meta_.coef_.round(4)))


# --------------------------------------------------------------------------
# 4. Metrics
# --------------------------------------------------------------------------
def smape(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.mean(2 * np.abs(p - y) / (np.abs(y) + np.abs(p) + 1e-9)) * 100)


def metrics(y, p):
    return dict(MAE=float(mean_absolute_error(y, p)),
                RMSE=float(mean_squared_error(y, p) ** 0.5),
                R2=float(r2_score(y, p)),
                sMAPE=smape(y, p))


def to_cases(log_pred):
    return np.clip(np.expm1(np.asarray(log_pred)), 0, None)


# --------------------------------------------------------------------------
# 5. Future forecasting (recursive, one month at a time)
# --------------------------------------------------------------------------
CLIM_COLS = ["rain", "tmax", "tmin", "hum"]


def make_climatology(hist: pd.DataFrame) -> pd.DataFrame:
    """Average rain/temperature/humidity per district and calendar month."""
    h = hist.assign(cal_month=hist["month_start"].dt.month)
    return h.groupby(["district", "cal_month"])[CLIM_COLS].mean()


def recursive_forecast(model, hist: pd.DataFrame, n_months: int, clim=None) -> pd.DataFrame:
    """Forecast `n_months` after the end of `hist`.

    Each predicted month is appended to the history and used to build the lag
    features of the next month. Future weather is unknown, so it is replaced
    by the district's historical average for that calendar month.
    """
    if clim is None:
        clim = make_climatology(hist)
    last = hist["month_start"].max()
    keep_from = last - pd.DateOffset(months=15)          # 12 lags + margin is enough
    work = hist[hist["month_start"] >= keep_from][BASE_COLS].copy()
    static = (hist.sort_values("month_start").groupby("district").tail(1)
                  [["district", "population", "lat", "lon", "coastal"]])
    out = []
    for h in range(1, n_months + 1):
        ms = last + pd.DateOffset(months=h)
        c = clim.xs(ms.month, level="cal_month").reset_index()
        new = static.merge(c, on="district")
        new["month_start"] = ms
        new["cases"] = np.nan
        work = pd.concat([work, new[BASE_COLS]], ignore_index=True)
        feat = add_features(work)
        rows = feat[feat["month_start"] == ms]
        pred = to_cases(model.predict(rows[FEATURES]))
        pmap = dict(zip(rows["district"], pred))
        mask = work["month_start"] == ms
        work.loc[mask, "cases"] = work.loc[mask, "district"].map(pmap)
        out.append(pd.DataFrame({"district": rows["district"].values, "month_start": ms,
                                 "horizon": h, "pred": pred}))
    return pd.concat(out, ignore_index=True)


Z80 = 1.2816  # z-value for an (approximate) 80% interval


def horizon_bucket(h: int) -> str:
    return "1-3" if h <= 3 else "4-6" if h <= 6 else "7-12" if h <= 12 else "13+"


def band(pred, h, sigma_tbl):
    """Approximate 80% range using the model's measured multi-step log-error."""
    s = sigma_tbl[horizon_bucket(int(h))]
    lo = (np.asarray(pred) + 1) * np.exp(-Z80 * s) - 1
    hi = (np.asarray(pred) + 1) * np.exp(Z80 * s) - 1
    return np.clip(lo, 0, None), hi
