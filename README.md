# Dengue Ensemble Dashboard (Sri Lanka)

An interactive Streamlit dashboard that forecasts monthly dengue cases, per district and nationwide, and compares five machine-learning models:

| Role | Model |
|---|---|
| Baseline (no tuning, no CV) | Linear Regression |
| Bagging | Random Forest |
| Boosting | XGBoost |
| Stacking | RF + XGBoost + Ridge + KNN, blended by a Ridge meta-learner |
| Hybrid | SARIMA (national time-series part) + Random Forest on the residuals |

## Dashboard pages
- **Overview**: key numbers, national trend, seasonality and top districts
- **Model Comparison**: metrics for all five models side by side (next month and 24 months ahead)
- **Past Predictions**: real vs predicted cases for any district and model
- **Future Forecast**: enter a year + month + region and get a prediction with an 80% range

## Run the dashboard
Full VS Code walkthrough: see [VSCODE_GUIDE.md](VSCODE_GUIDE.md). Quick version, from this folder:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app/app.py
```

## Re-train the models
Download the dataset (Kaggle: `tharinduheshan/dengue-research-2006-2026`) and save it as
`data/raw/dengue_srilanka_full_2006_2025.csv`. Then open `notebooks/dengue_ensemble.ipynb` and run all cells.
It writes `data/processed/`, `artifacts/` and `models/` (about 3 minutes). Shared code is in `src/pipeline.py`.

## Project structure
```
app/         Streamlit dashboard
src/         pipeline.py: features, time-aware stacking, SARIMA+RF hybrid, recursive forecasting
notebooks/   training and comparison
models/      the five deployed models (refit on all data)
artifacts/   metrics and predictions
data/        raw Kaggle CSV (not committed) and the processed monthly panel
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

## AI Usage Acknowledgment
In line with the module's GenAI policy, I declare the following use of AI tools:

- **Tool used:** Claude (Anthropic), via Claude Code
- **Where and how it was used:**
  - Generating the Python code for data preparation, feature engineering, model training (Linear Regression baseline, Random Forest, XGBoost, Stacking, SARIMA + Random Forest hybrid), hyperparameter tuning, evaluation, forecasting and charts
  - Building the Streamlit dashboard (layout, styling and pages)
  - Drafting the markdown explanations, including the results interpretation and the demo script
  - Debugging errors encountered while running the notebook and the dashboard
