# Dengue Ensemble Dashboard (Sri Lanka)

An interactive Streamlit dashboard that forecasts monthly dengue cases, per district and nationwide, and compares four machine-learning models.

## Models

| Role | Model |
|---|---|
| Baseline (no tuning, no CV) | Linear Regression |
| Bagging | Random Forest |
| Boosting | XGBoost |
| Stacking | RF + XGBoost + Ridge + KNN, blended by a Ridge meta-learner |

## Dashboard pages
- **Overview**: the data and project summary
- **Model Comparison**: metrics for all four models side by side
- **Past Predictions**: real vs predicted cases
- **Future Forecast**: pick a year, month and region to get a forecast from every model

## Quick start
Requires Python 3.10 or newer. From the project folder:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app/app.py
```

The dashboard opens at http://localhost:8501. The trained models and processed data are included, so no training is needed to run it.

> `requirements.txt` pins exact versions. The saved models (`models/*.joblib`) only load reliably with the same scikit-learn and XGBoost versions they were trained with.

## Re-train the models
1. Download **Dengue_Research_2006-2026** from Kaggle (`tharinduheshan/dengue-research-2006-2026`).
2. Save the CSV as `data/raw/dengue_srilanka_full_2006_2025.csv`.
3. Open `notebooks/dengue_ensemble.ipynb`, select the `.venv` kernel and run all cells (about 2 minutes).

The notebook rewrites `data/processed/`, `artifacts/` and `models/`. Restart the dashboard afterwards to see the new results.

## Project structure
```
app/         Streamlit dashboard
src/         pipeline.py: features, time-aware stacking, recursive forecasting
notebooks/   training and model comparison
models/      the four deployed models (refit on all data)
artifacts/   metrics and predictions used by the dashboard
data/
  raw/       Kaggle CSV (not committed)
  processed/ monthly_panel.csv
.streamlit/  dashboard theme and settings
```

## Troubleshooting
| Problem | Fix |
|---|---|
| `streamlit` is not recognised | Activate the venv: `.venv\Scripts\Activate.ps1` |
| `ModuleNotFoundError: pipeline` | Run commands from the project root folder |
| Port 8501 already in use | `streamlit run app/app.py --server.port 8600` |
| Dashboard looks dark | Run it from the project root so `.streamlit/config.toml` is picked up |
| Notebook says `No module named sklearn` | Select the `.venv` kernel |
| `FileNotFoundError` for the CSV | Download the dataset (see Re-train the models) |

## Data source
Dengue_Research_2006-2026 on Kaggle (`tharinduheshan/dengue-research-2006-2026`).
