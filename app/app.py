"""Dengue Ensemble Dashboard (Streamlit).

Run from the project folder:  streamlit run app/app.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import forecast as F

# --------------------------------------------------------------------------
# Page setup + look
# --------------------------------------------------------------------------
st.set_page_config(page_title="Dengue Ensemble Dashboard", page_icon="🦟", layout="wide")

ACCENT = "#2563eb"
COLORS = {"baseline": "#94a3b8", "rf": "#3b82f6", "xgb": "#38bdf8", "stack": "#1e3a8a", "hybrid": "#f59e0b"}
# high-contrast colours + dash styles for the line charts (real cases stay black)
LINE_STYLE = {"baseline": ("#f97316", "dash"), "rf": ("#0ea5e9", "solid"),
              "xgb": ("#16a34a", "dot"), "stack": ("#7c3aed", "solid"), "hybrid": ("#e11d48", "dashdot")}
INK = "#0f172a"
MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]

st.markdown(f"""
<style>
  .block-container {{ padding-top: 2rem; max-width: 1250px; }}
  h1, h2, h3 {{ letter-spacing: -0.02em; }}
  .hero {{ background: linear-gradient(135deg, #1d4ed8 0%, #1e40af 55%, #172554 100%);
          border-radius: 18px; padding: 26px 30px; color: white; margin-bottom: 18px; }}
  .hero h1 {{ color: white; margin: 0 0 4px 0; font-size: 1.9rem; }}
  .hero p {{ color: #dbeafe; margin: 0; font-size: 1.0rem; }}
  .card {{ background: #ffffff; border: 1px solid #e2e8f0; border-radius: 14px; padding: 16px 18px;
          box-shadow: 0 1px 2px rgba(15,23,42,.04); height: 100%; }}
  .card .lbl {{ color: #64748b; font-size: .78rem; text-transform: uppercase; letter-spacing: .06em; }}
  .card .val {{ color: {INK}; font-size: 1.85rem; font-weight: 700; line-height: 1.25; }}
  .card .sub {{ color: #64748b; font-size: .85rem; }}
  .card.big {{ background: #eff6ff; border-color: #bfdbfe; }}
  .card.big .val {{ color: {ACCENT}; font-size: 2.6rem; }}
  .note {{ background: #f8fafc; border-left: 4px solid {ACCENT}; padding: 12px 16px; border-radius: 8px;
          color: #334155; font-size: .93rem; }}
  .warn {{ background: #fffbeb; border-left: 4px solid #f59e0b; padding: 12px 16px; border-radius: 8px;
          color: #78350f; font-size: .93rem; }}
  .pill {{ display:inline-block; padding: 2px 10px; border-radius: 999px; font-size: .75rem; font-weight: 600;
          background: #dbeafe; color: #1e40af; }}

  /* ---- coloured top bar ---- */
  header[data-testid="stHeader"] {{ background: #000000;
          border-bottom: 3px solid #2563eb; box-shadow: 0 2px 8px rgba(15,23,42,.25); }}
  header[data-testid="stHeader"] *, header[data-testid="stHeader"] button {{ color: #ffffff !important; }}
  header[data-testid="stHeader"] svg {{ fill: #ffffff; }}
  .block-container {{ padding-top: 4.5rem; }}

  /* ---- dark blue sidebar ---- */
  section[data-testid="stSidebar"] {{ background: linear-gradient(180deg, #0b1b45 0%, #0a1a40 55%, #12307a 100%);
          border-right: 2px solid rgba(255,255,255,.55); }}
  section[data-testid="stSidebar"] * {{ color: #cbd5e1; }}
  section[data-testid="stSidebar"] hr {{ border-color: rgba(148,163,184,.25); }}
  .brand {{ display:flex; align-items:center; gap:12px; padding: 6px 4px 18px 4px; }}
  .brand .logo {{ width:48px; height:48px; border-radius:50%; background:#b91c1c; display:flex;
          align-items:center; justify-content:center; font-size:26px; flex:none; }}
  .brand .t1 {{ color:#fff !important; font-weight:800; font-size:1.25rem; line-height:1.1; }}
  .brand .t2 {{ color:#e2e8f0 !important; font-size:1rem; }}
  section[data-testid="stSidebar"] div[role="radiogroup"] {{ gap: 6px; }}
  section[data-testid="stSidebar"] div[role="radiogroup"] label {{ padding: 11px 14px; border-radius: 12px;
          width: 100%; cursor: pointer; transition: background .15s; }}
  section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {{ background: rgba(255,255,255,.08); }}
  section[data-testid="stSidebar"] div[role="radiogroup"] label > div > div:first-child {{ display:none; }}
  section[data-testid="stSidebar"] div[role="radiogroup"] label p {{ font-size: .98rem; font-weight: 500;
          display:flex; align-items:center; gap: 14px; margin:0; }}
  section[data-testid="stSidebar"] div[role="radiogroup"] label p::before {{ content:""; width:20px; height:20px;
          background-color:#e2e8f0; -webkit-mask-size:contain; mask-size:contain; -webkit-mask-repeat:no-repeat;
          mask-repeat:no-repeat; -webkit-mask-position:center; mask-position:center; flex:none; }}
  section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {{ background: #2f6df0;
          box-shadow: 0 4px 14px rgba(47,109,240,.45); }}
  section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) p {{ color:#fff; font-weight:600; }}
  section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) p::before {{ background-color:#fff; }}
  section[data-testid="stSidebar"] div[role="radiogroup"] > div:nth-of-type(1) label p::before {{ -webkit-mask-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'><path d='M3 13h8V3H3v10zm0 8h8v-6H3v6zm10 0h8V11h-8v10zm0-18v6h8V3h-8z'/></svg>"); mask-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'><path d='M3 13h8V3H3v10zm0 8h8v-6H3v6zm10 0h8V11h-8v10zm0-18v6h8V3h-8z'/></svg>"); }}
  section[data-testid="stSidebar"] div[role="radiogroup"] > div:nth-of-type(2) label p::before {{ -webkit-mask-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'><path d='M4 18h16v2H4v-2zm1-3l4-5 4 3 6-8 1.6 1.2L13.4 15 9.5 12 6.6 15.8 5 15z'/></svg>"); mask-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'><path d='M4 18h16v2H4v-2zm1-3l4-5 4 3 6-8 1.6 1.2L13.4 15 9.5 12 6.6 15.8 5 15z'/></svg>"); }}
  section[data-testid="stSidebar"] div[role="radiogroup"] > div:nth-of-type(3) label p::before {{ -webkit-mask-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'><path d='M13 3a9 9 0 100 18 9 9 0 000-18zm1 9.4l3.5 2.1-.8 1.3L12 13V7h2v5.4z'/></svg>"); mask-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'><path d='M13 3a9 9 0 100 18 9 9 0 000-18zm1 9.4l3.5 2.1-.8 1.3L12 13V7h2v5.4z'/></svg>"); }}
  section[data-testid="stSidebar"] div[role="radiogroup"] > div:nth-of-type(4) label p::before {{ -webkit-mask-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'><path d='M12 2l3 7 7 .6-5.3 4.6 1.6 7.1L12 17.8 5.7 21.3l1.6-7.1L2 9.6 9 9z'/></svg>"); mask-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'><path d='M12 2l3 7 7 .6-5.3 4.6 1.6 7.1L12 17.8 5.7 21.3l1.6-7.1L2 9.6 9 9z'/></svg>"); }}
</style>
""", unsafe_allow_html=True)


def card(label, value, sub="", big=False):
    cls = "card big" if big else "card"
    return f'<div class="{cls}"><div class="lbl">{label}</div><div class="val">{value}</div><div class="sub">{sub}</div></div>'


def fmt(x):
    return f"{x:,.0f}"


def style_fig(fig, height=380):
    fig.update_layout(template="plotly_white", height=height, margin=dict(l=10, r=10, t=85, b=10),
                      title=dict(x=0, xanchor="left", y=0.96, yanchor="top"),
                      legend=dict(orientation="h", y=1.0, yanchor="bottom", x=0), hovermode="x unified",
                      font=dict(family="Inter, Segoe UI, sans-serif", size=13))
    return fig


# --------------------------------------------------------------------------
# Cached data / models
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_data():
    return F.load_metrics(), F.load_panel(), F.load_onestep(), F.load_backtest()


@st.cache_data(show_spinner="Forecasting the next 36 months with all five models (first load only)...")
def get_forecasts():
    metrics, panel, _, _ = get_data()
    return F.forecast_all(F.load_models(), panel)


metrics, panel, onestep, backtest = get_data()
LABEL = metrics["labels"]
SHORT = {"baseline": "Baseline", "rf": "Random Forest", "xgb": "XGBoost", "stack": "Stacking", "hybrid": "Hybrid (SARIMA+RF)"}
BEST = metrics["best_model"]
DISTRICTS = sorted(panel["district"].unique())
REGIONS = [F.ALL] + DISTRICTS
DATA_END = pd.Timestamp(metrics["data_end"])
CUTOFF = pd.Timestamp(metrics["cutoff"])
nat = panel.groupby("month_start")["cases"].sum()

# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="brand"><div class="logo">🦟</div><div><div class="t1">Dengue<br>Forecast</div>'
                '<div class="t2">Sri Lanka</div></div></div>', unsafe_allow_html=True)
    page = st.radio("Navigate", ["Overview", "Model Comparison", "Past Predictions", "Future Forecast"],
                    label_visibility="collapsed")
    st.markdown("---")
    st.caption(f"Data: {pd.Timestamp(metrics['data_start']):%b %Y} to {DATA_END:%b %Y} · 25 districts")
    st.caption(f"Test period: {(CUTOFF + pd.DateOffset(months=1)):%b %Y} to {DATA_END:%b %Y}")
    st.caption("Models: Linear Regression (baseline), Random Forest (bagging), XGBoost (boosting), Stacking, SARIMA + Random Forest (hybrid).")

st.markdown(
    '<div class="hero"><h1>Dengue Cases in Sri Lanka</h1>'
    '<p>Ensemble learning dashboard: compare bagging, boosting and stacking, review past predictions, '
    'and forecast any future month.</p></div>', unsafe_allow_html=True)

# ==========================================================================
# PAGE 1: Overview
# ==========================================================================
if page == "Overview":
    last12 = nat.iloc[-12:].sum()
    prev12 = nat.iloc[-24:-12].sum()
    peak_m = nat.idxmax()
    c = st.columns(4)
    c[0].markdown(card("Cases, last 12 months", fmt(last12),
                       f"{(last12 / prev12 - 1) * 100:+.0f}% vs the 12 months before"), unsafe_allow_html=True)
    c[1].markdown(card("Worst month on record", f"{peak_m:%b %Y}", f"{fmt(nat.max())} cases nationwide"),
                  unsafe_allow_html=True)
    c[2].markdown(card("Months of history", f"{len(nat)}", f"{nat.index.min():%Y} to {nat.index.max():%Y}"),
                  unsafe_allow_html=True)
    c[3].markdown(card("Best model", SHORT[BEST], "lowest test error (RMSE)"), unsafe_allow_html=True)
    st.write("")

    fig = go.Figure(go.Scatter(x=nat.index, y=nat.values, mode="lines", line=dict(color=ACCENT, width=2),
                               fill="tozeroy", fillcolor="rgba(37,99,235,.10)", name="Cases"))
    fig.update_layout(title="National dengue cases per month")
    st.plotly_chart(style_fig(fig, 340), width="stretch")

    a, b = st.columns(2)
    d = panel.assign(m=panel["month_start"].dt.month, y=panel["month_start"].dt.year)
    season = d.groupby(["y", "m"])["cases"].sum().groupby("m").mean()
    fig = go.Figure(go.Bar(x=[m[:3] for m in MONTHS], y=season.values, marker_color=ACCENT))
    fig.update_layout(title="Seasonality: average cases by calendar month")
    a.plotly_chart(style_fig(fig, 320), width="stretch")
    top = panel.groupby("district")["cases"].sum().sort_values().tail(10)
    fig = go.Figure(go.Bar(x=top.values, y=top.index, orientation="h", marker_color="#60a5fa"))
    fig.update_layout(title="Top 10 districts by total cases")
    b.plotly_chart(style_fig(fig, 320), width="stretch")

    st.markdown("#### How it works")
    h = st.columns(3)
    h[0].markdown(card("1 · Data", "Weekly → monthly",
                       "25 districts, cases + rainfall, temperature, humidity, population"), unsafe_allow_html=True)
    h[1].markdown(card("2 · Models", "5 compared",
                       "Linear baseline, Random Forest (bagging), XGBoost (boosting), Stacking and a SARIMA + Random Forest hybrid"),
                  unsafe_allow_html=True)
    h[2].markdown(card("3 · Forecast", "Month by month",
                       "Each predicted month feeds the next, with a measured uncertainty range"),
                  unsafe_allow_html=True)

# ==========================================================================
# PAGE 2: Model comparison
# ==========================================================================
elif page == "Model Comparison":
    ctl = st.columns([1.2, 1.2, 2])
    horizon = ctl[0].radio("Forecast type", ["Next month", "24 months ahead"], horizontal=True,
                           help="Next month: each prediction uses the real previous months. "
                                "24 months ahead: a blind recursive forecast, like the future-forecast page.")
    level = ctl[1].radio("Level", ["District-month", "National total"], horizontal=True)
    src = metrics["one_step"] if horizon == "Next month" else metrics["multi_step"]
    lvl = "district" if level == "District-month" else "national"
    df = pd.DataFrame({LABEL[k]: v[lvl] for k, v in src.items()}).T[["MAE", "RMSE", "R2", "sMAPE"]]
    winner = df["RMSE"].idxmin()

    st.markdown(f'<div class="note"><b>Winner for this view (lowest RMSE): {winner}.</b> '
                f'Test period: {metrics["test_months"]} months '
                f'({(CUTOFF + pd.DateOffset(months=1)):%b %Y} to {DATA_END:%b %Y}), '
                f'never seen during training.</div>', unsafe_allow_html=True)
    st.write("")

    sty = (df.style.format({"MAE": "{:,.1f}", "RMSE": "{:,.1f}", "R2": "{:.3f}", "sMAPE": "{:.1f}%"})
           .highlight_min(subset=["MAE", "RMSE", "sMAPE"], color="#dbeafe")
           .highlight_max(subset=["R2"], color="#dbeafe"))
    st.dataframe(sty, width="stretch")
    st.caption("MAE / RMSE are in cases per " + ("district-month" if lvl == "district" else "month (all Sri Lanka)")
               + ". Lower is better, except R² (higher is better). Best value in each column is highlighted.")

    fig = make_subplots(rows=1, cols=4, subplot_titles=["MAE", "RMSE", "R²", "sMAPE (%)"])
    names = list(df.index)
    cols = [COLORS[k] for k in src.keys()]
    for i, m in enumerate(["MAE", "RMSE", "R2", "sMAPE"], start=1):
        fig.add_trace(go.Bar(x=[SHORT[k].split(" (")[0] for k in src.keys()], y=df[m].values, marker_color=cols,
                             text=[f"{v:,.2f}" if m == "R2" else f"{v:,.0f}" for v in df[m].values],
                             textposition="outside", showlegend=False), row=1, col=i)
    fig.update_layout(hovermode="closest")
    st.plotly_chart(style_fig(fig, 340), width="stretch")

    with st.expander("What is each model?"):
        st.markdown(
            "- **Baseline: Linear Regression.** No tuning, no cross-validation.\n"
            "- **Bagging: Random Forest.** Many deep trees on bootstrap samples, averaged.\n"
            "- **Boosting: XGBoost.** Shallow trees added one by one, each fixing the previous errors.\n"
            "- **Stacking.** Random Forest, XGBoost, Ridge and KNN blended by a Ridge meta-learner.\n"
            "- **Hybrid: SARIMA + Random Forest.** SARIMA models the national trend and seasonality; a Random Forest then learns the remaining per-district error.")

    with st.expander("Feature importance, stacking weights and tuned settings"):
        a, b = st.columns(2)
        for col, k in ((a, "rf"), (b, "xgb")):
            s = pd.Series(metrics["importance"][k]).sort_values().tail(10)
            fig = go.Figure(go.Bar(x=s.values, y=s.index, orientation="h", marker_color=COLORS[k]))
            fig.update_layout(title=f"Top features: {SHORT[k]}", hovermode="closest")
            col.plotly_chart(style_fig(fig, 340), width="stretch")
        ws = pd.Series(metrics["stack_weights"])
        fig = go.Figure(go.Bar(x=[x.upper() for x in ws.index], y=ws.values, marker_color=COLORS["stack"]))
        fig.update_layout(title="Stacking meta-learner weights", hovermode="closest")
        st.plotly_chart(style_fig(fig, 280), width="stretch")
        st.json({"Random Forest (tuned)": metrics["best_params"]["rf"],
                 "XGBoost (tuned)": metrics["best_params"]["xgb"],
                 "Baseline": "Linear Regression, default settings, no tuning, no cross-validation"})

# ==========================================================================
# PAGE 3: Past predictions
# ==========================================================================
elif page == "Past Predictions":
    ctl = st.columns([1.2, 2, 1.6])
    region = ctl[0].selectbox("Region", REGIONS)
    chosen = ctl[1].multiselect("Models", list(F.MODEL_KEYS), default=["baseline", BEST],
                                format_func=lambda k: SHORT[k])
    mode = ctl[2].radio("Prediction type", ["Next month", "24-month blind forecast"],
                        help="Next month: uses the real previous months. "
                             "24-month blind forecast: starts at the cut-off and feeds on its own predictions.")
    actual = F.region_actual(panel, region)

    if mode == "Next month":
        lo, hi = actual.index.min(), actual.index.max()
        start_default = pd.Timestamp("2016-01-01")
        rng = st.slider("Date range", min_value=lo.to_pydatetime(), max_value=hi.to_pydatetime(),
                        value=(start_default.to_pydatetime(), hi.to_pydatetime()), format="MMM YYYY")
        d = onestep if region == F.ALL else onestep[onestep["district"] == region]
        pred = d.groupby("month_start")[list(F.MODEL_KEYS)].sum()
        a = actual[(actual.index >= rng[0]) & (actual.index <= rng[1])]
        shade = (CUTOFF + pd.DateOffset(months=1), hi)
    else:
        bt = backtest if region == F.ALL else backtest[backtest["district"] == region]
        pred = bt.pivot_table(index="month_start", columns="model", values="pred", aggfunc="sum")
        a = actual[actual.index > CUTOFF]
        shade = None

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=a.index, y=a.values, name="Real cases", mode="lines+markers",
                             line=dict(color=INK, width=3), marker=dict(size=4)))
    for k in chosen:
        p = pred[k].reindex(a.index).dropna()
        fig.add_trace(go.Scatter(x=p.index, y=p.values, name=SHORT[k], mode="lines",
                                 line=dict(color=LINE_STYLE[k][0], width=2.4, dash=LINE_STYLE[k][1])))
    if shade:
        fig.add_vrect(x0=shade[0], x1=shade[1], fillcolor="#f59e0b", opacity=0.10, line_width=0,
                      annotation_text="test period (unseen)", annotation_position="top left")
    fig.update_layout(title=f"Real vs predicted dengue cases: {region}", yaxis_title="cases per month")
    st.plotly_chart(style_fig(fig, 460), width="stretch")

    if mode == "Next month":
        st.caption("Before the shaded area the models are drawn over data they were trained on, so that part looks "
                   "better than a real forecast would. Judge the models on the shaded test period.")
    else:
        st.caption("Blind forecast: only data up to the cut-off was known. Each month is predicted from the model's "
                   "own earlier predictions, with average weather. This is how the Future Forecast page works.")

    # error table for the test period
    tcut = a.index > CUTOFF if mode == "Next month" else a.index == a.index
    rows = []
    for k in chosen:
        p = pred[k].reindex(a.index)
        m = tcut & p.notna().values
        if m.sum():
            err = p[m] - a[m]
            rows.append({"Model": SHORT[k], "MAE (cases/month)": err.abs().mean(),
                         "RMSE": float(np.sqrt((err ** 2).mean())), "Test months": int(m.sum())})
    if rows:
        st.markdown(f"**Accuracy on the test period for {region}**")
        st.dataframe(pd.DataFrame(rows).style.format({"MAE (cases/month)": "{:,.0f}", "RMSE": "{:,.0f}"}),
                     width="stretch", hide_index=True)

# ==========================================================================
# PAGE 4: Future forecast
# ==========================================================================
else:
    fc = get_forecasts()
    first_future = DATA_END + pd.DateOffset(months=1)
    last_future = DATA_END + pd.DateOffset(months=F.MAX_HORIZON)

    st.markdown("#### Pick a month to forecast")
    c = st.columns([1, 1, 1.5, 1.8])
    year = c[0].selectbox("Year", list(range(2008, last_future.year + 1)),
                          index=list(range(2008, last_future.year + 1)).index(2027))
    month = c[1].selectbox("Month", MONTHS, index=0)
    region = c[2].selectbox("Region", REGIONS)
    model = c[3].selectbox("Model", list(F.MODEL_KEYS), index=list(F.MODEL_KEYS).index(BEST),
                           format_func=lambda k: LABEL[k])
    target = pd.Timestamp(year=int(year), month=MONTHS.index(month) + 1, day=1)
    tname = f"{month} {year}"

    actual = F.region_actual(panel, region)
    hist_months = actual.index

    # typical value for that calendar month, from the last 5 complete years of data
    last5 = actual[actual.index > DATA_END - pd.DateOffset(years=5)]
    typical = last5[last5.index.month == target.month].mean()

    if target > last_future:
        st.markdown(f'<div class="warn">Forecasts are limited to 36 months after the last month of data. '
                    f'Please choose a month up to <b>{last_future:%B %Y}</b>.</div>', unsafe_allow_html=True)
    elif target <= DATA_END:
        # past month: show what really happened and what the model said
        real = actual.get(target, np.nan)
        d = onestep if region == F.ALL else onestep[onestep["district"] == region]
        pr = d.groupby("month_start")[model].sum().get(target, np.nan)
        st.markdown(f'<div class="note"><b>{tname} is in the past</b> (data runs to {DATA_END:%B %Y}), so here is what '
                    f'really happened. Pick a month from <b>{first_future:%B %Y}</b> onwards to get a forecast.</div>',
                    unsafe_allow_html=True)
        st.write("")
        cc = st.columns(3)
        cc[0].markdown(card(f"Real cases in {tname}", fmt(real), region, big=True), unsafe_allow_html=True)
        cc[1].markdown(card(f"{SHORT[model]} next-month prediction", fmt(pr) if pr == pr else "n/a",
                            "made with the real previous months" if pr == pr else "no prediction before 2008"),
                       unsafe_allow_html=True)
        cc[2].markdown(card(f"Typical {month}", fmt(typical), "average of the last 5 years"), unsafe_allow_html=True)
    else:
        series = F.region_series(fc, model, region, metrics["sigma"])
        row = series[series["month_start"] == target].iloc[0]
        h = int(row["horizon"])
        pred, lo, hi = row["pred"], row["low"], row["high"]
        delta = (pred / typical - 1) * 100 if typical else np.nan

        cc = st.columns([1.6, 1.2, 1.2])
        cc[0].markdown(card(f"Predicted dengue cases · {tname}", fmt(pred),
                            f"{region} · {SHORT[model]} model", big=True), unsafe_allow_html=True)
        cc[1].markdown(card("Likely range (80%)", f"{fmt(lo)} to {fmt(hi)}",
                            f"{h} month{'s' if h > 1 else ''} ahead"), unsafe_allow_html=True)
        cc[2].markdown(card(f"vs typical {month}", f"{delta:+.0f}%",
                            f"typical = {fmt(typical)} (last 5 yrs)"), unsafe_allow_html=True)
        st.write("")

        left, right = st.columns([1.7, 1.3])
        with left:
            hist = actual[actual.index > DATA_END - pd.DateOffset(months=36)]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=hist.index, y=hist.values, name="Real cases", mode="lines",
                                     line=dict(color=INK, width=2.5)))
            s2 = series[series["month_start"] <= target]
            fig.add_trace(go.Scatter(x=s2["month_start"], y=s2["high"], mode="lines", line=dict(width=0),
                                     showlegend=False, hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=s2["month_start"], y=s2["low"], mode="lines", line=dict(width=0),
                                     fill="tonexty", fillcolor="rgba(37,99,235,.18)", name="80% range",
                                     hoverinfo="skip"))
            xs = pd.concat([pd.Series([DATA_END]), s2["month_start"]])
            ys = pd.concat([pd.Series([actual.iloc[-1]]), s2["pred"]])
            fig.add_trace(go.Scatter(x=xs, y=ys, name="Forecast", mode="lines",
                                     line=dict(color=COLORS[model], width=2.5, dash="dot")))
            fig.add_trace(go.Scatter(x=[target], y=[pred], name=tname, mode="markers",
                                     marker=dict(size=14, color="#ef4444", line=dict(color="white", width=2))))
            fig.update_layout(title=f"History and forecast: {region}", yaxis_title="cases per month")
            st.plotly_chart(style_fig(fig, 420), width="stretch")
        with right:
            st.markdown(f"**All five models, {tname}**")
            comp = []
            for k in F.MODEL_KEYS:
                r = F.region_series(fc, k, region, metrics["sigma"])
                r = r[r["month_start"] == target].iloc[0]
                comp.append({"Model": SHORT[k], "Cases": r["pred"], "Low": r["low"], "High": r["high"]})
            cdf = pd.DataFrame(comp)
            st.dataframe(cdf.style.format({"Cases": "{:,.0f}", "Low": "{:,.0f}", "High": "{:,.0f}"}),
                         width="stretch", hide_index=True)
            if region == F.ALL:
                dd = fc[(fc["model"] == model) & (fc["month_start"] == target)].nlargest(8, "pred").iloc[::-1]
                fig = go.Figure(go.Bar(x=dd["pred"], y=dd["district"], orientation="h", marker_color=COLORS[model]))
                fig.update_layout(title="Top districts", hovermode="closest")
                st.plotly_chart(style_fig(fig, 300), width="stretch")

        if h > 12:
            st.markdown('<div class="warn"><b>Long-range forecast.</b> More than 12 months ahead the model mainly '
                        'reproduces the usual seasonal pattern and recent trend, so the range is wide.</div>',
                        unsafe_allow_html=True)

    st.write("")
    with st.expander("How is this forecast made, and how far can I trust it?"):
        sg = metrics["sigma"][model]["national"]
        st.markdown(f"""
* The model predicts **one month at a time**. Each predicted month becomes the "last month" input for the next one,
  from the end of the data ({DATA_END:%B %Y}) up to your chosen month.
* **Future weather is unknown**, so rainfall, temperature and humidity are replaced by each district's average for
  that calendar month. The forecast therefore cannot anticipate an unusually wet or dry season.
* The **80% range** is measured, not guessed: from 12 different start dates in the test period we forecast
  12 months ahead and recorded the errors. For the selected model the typical national error is about
  ±{(np.exp(1.2816 * sg['1-3']) - 1) * 100:.0f}% at 1 to 3 months, ±{(np.exp(1.2816 * sg['4-6']) - 1) * 100:.0f}% at 4 to 6 months and
  ±{(np.exp(1.2816 * sg['7-12']) - 1) * 100:.0f}% at 7 to 12 months.
* Models capture **seasonality and the current trend, not unforeseeable outbreaks** (every model missed the
  Dec 2023 surge when tested).
""")
