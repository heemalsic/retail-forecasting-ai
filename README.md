# Retail Forecasting AI

![Alt Text](https://github.com/heemalsic/retail-forecasting-ai/blob/main/images/dashboard.png)
End-to-end retail demand forecasting project with:
- data preprocessing + feature engineering,
- XGBoost model training,
- FastAPI prediction service,
- Streamlit dashboard for forecasting + inventory simulation,
- optional Gemini-powered business insights.

## Project Overview

This repository predicts weekly retail sales and simulates inventory behavior for selected store/department combinations.

The workflow is:
1. Merge and clean raw Walmart retail datasets.
2. Create time-series features (lag + rolling statistics).
3. Train an `XGBRegressor` model.
4. Serve predictions via API and visualize outcomes in a dashboard.
5. Generate natural-language recommendations using Gemini (optional).

## Dataset

The model is trained on the **Walmart Sales Forecast** dataset from Kaggle:

- https://www.kaggle.com/datasets/aslanahmedov/walmart-sales-forecast

Expected raw files in `data/raw/`:
- `train.csv`
- `test.csv`
- `features.csv`
- `stores.csv`

## Repository Structure

```text
retail-forecasting-ai/
├── api/
│   └── app.py                  # FastAPI app with / and /predict
├── dashboard/
│   └── app.py                  # Streamlit dashboard UI
├── data/
│   ├── raw/
│   │   ├── train.csv
│   │   ├── test.csv
│   │   ├── features.csv
│   │   └── stores.csv
│   └── processed/
│       └── final_dataset.csv   # Cleaned + engineered dataset
├── models/
│   └── xgb_model.pkl           # Trained model artifact
├── src/
│   ├── data_preprocessing.py   # Merge + clean raw data
│   ├── feature_engineering.py  # Temporal + lag/rolling features
│   ├── train.py                # Model training script
│   ├── predict.py              # Programmatic prediction helper
│   ├── inventory.py            # Inventory simulation logic
│   └── llm_insights.py         # Gemini insights generation
├── requirements.txt
└── README.md
```

## Tech Stack

- Python
- pandas, scikit-learn, xgboost, joblib
- FastAPI + Uvicorn
- Streamlit
- google-genai (optional, for LLM insights)

## Setup

### 1) Create and activate a virtual environment

#### Windows (PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

#### macOS/Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies

`requirements.txt` is currently empty, so install packages manually:

```powershell
pip install pandas scikit-learn xgboost joblib fastapi uvicorn streamlit google-genai
```

## ⚠️ Gemini API Key (Required for AI Insights)

If you want the **Generate Insights** button in the dashboard to work, set `GEMINI_API_KEY` in your terminal session before running the app.

### Windows PowerShell (current session)
```powershell
$env:GEMINI_API_KEY="your_api_key_here"
```

### Windows PowerShell (persist for future sessions)
```powershell
setx GEMINI_API_KEY "your_api_key_here"
```

### macOS/Linux (bash/zsh)
```bash
export GEMINI_API_KEY="your_api_key_here"
```

### Quick check
```powershell
echo $env:GEMINI_API_KEY
```

> Note: If this variable is not set, the project still runs, but LLM insight generation will return a warning.

## Data Pipeline

Run scripts from the repository root:

### 1) Preprocess raw data
```powershell
python src/data_preprocessing.py
```

### 2) Feature engineering
```powershell
python src/feature_engineering.py
```

### 3) Train model
```powershell
python src/train.py
```

This creates/updates `models/xgb_model.pkl`.

## Run the FastAPI Service

```powershell
uvicorn api.app:app --reload
```

Default URL: `http://127.0.0.1:8000`

### Endpoints
- `GET /` → health/message endpoint
- `POST /predict` → returns `predicted_sales`

### Example request body for `POST /predict`
```json
{
	"Store": 1,
	"Dept": 1,
	"Temperature": 42.31,
	"Fuel_Price": 2.57,
	"CPI": 211.10,
	"Unemployment": 8.10,
	"IsHoliday": 0,
	"Size": 151315,
	"year": 2012,
	"month": 2,
	"week": 6,
	"lag_1": 24924.5,
	"lag_4": 23120.0,
	"rolling_mean_4": 24080.2
}
```

## Run the Streamlit Dashboard

```powershell
python -m streamlit run dashboard/app.py
```

The dashboard supports:
- sample data fallback (`data/processed/final_dataset.csv`),
- CSV upload,
- store/department filtering,
- forecast vs stock visualization,
- inventory risk alerts,
- optional Gemini-generated insights.

## Scripts Summary

- `src/data_preprocessing.py`: merges `train.csv`, `features.csv`, `stores.csv`; handles holiday column cleanup.
- `src/feature_engineering.py`: creates `year`, `month`, `week`, `lag_1`, `lag_4`, `rolling_mean_4`.
- `src/train.py`: trains `XGBRegressor`, prints MAE, saves model.
- `src/predict.py`: lightweight helper for model inference.
- `src/inventory.py`: stock depletion + restock simulation.
- `src/llm_insights.py`: Gemini API call for concise recommendations.

## Troubleshooting

- **`uvicorn` or `streamlit` command fails**: install missing packages in the active virtual environment.
- **Model load error (`models/xgb_model.pkl`)**: run preprocessing + feature engineering + training scripts first.
- **LLM insights error**: verify `google-genai` installation and `GEMINI_API_KEY` in the same terminal session.
- **Missing columns in dashboard**: ensure uploaded CSV matches expected schema and includes date/sales history needed for lag features.

## Future Improvements

- Add `requirements.txt` with pinned versions.
- Add automated tests for preprocessing, features, and API schema.
- Add model versioning/experiment tracking.
- Containerize API + dashboard with Docker.

