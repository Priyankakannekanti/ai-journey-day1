# California House Price Prediction

End-to-end machine learning pipeline using Scikit-learn on the California Housing dataset (20,640 records) to predict median house values.

## Workflow
1. Load California Housing dataset (`fetch_california_housing`)
2. Train-test split (80/20) and feature scaling (`StandardScaler`)
3. Train `RandomForestRegressor` and evaluate using MAE, RMSE, and R² score
4. Save trained model and scaler locally via `joblib` (excluded from Git via `.gitignore`)

## How to Run Locally
```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python house-prediction-1.py