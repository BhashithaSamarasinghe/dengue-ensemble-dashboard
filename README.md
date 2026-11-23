# Dengue Ensemble Dashboard (Sri Lanka)

Forecasts monthly dengue cases (per district and nationwide) and compares four models:

| Role | Model |
|---|---|
| Baseline (no tuning, no CV) | Linear Regression |
| Bagging | Random Forest |
| Boosting | XGBoost |
| Stacking | RF + XGBoost + Ridge + KNN, blended by a Ridge meta-learner |

## Run the dashboard
Full VS Code walkthrough: see [VSCODE_GUIDE.md](VSCODE_GUIDE.md). Quick version, from this folder:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app/app.py
```

Pages: Overview, Model Comparison, Past Predictions (real vs predicted), Future Forecast (enter year + month + region).

## Re-train everything
Open `notebooks/dengue_ensemble.ipynb` and run all cells. It writes `data/processed/`, `artifacts/` and `models/`
(about 2 minutes). Shared code is in `src/pipeline.py`.

## Folders
- `data/raw/` the Kaggle CSV (not committed), `data/processed/monthly_panel.csv`
- `src/pipeline.py` features, time-aware stacking, recursive forecasting
- `notebooks/` training and comparison
- `models/` four deployed models (refit on all data), `artifacts/` metrics and predictions
- `app/` Streamlit dashboard
