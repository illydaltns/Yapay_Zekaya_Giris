"""
evaluate_clean_comparison.py

Amaç:
- Volatility içermeyen Logistic Regression (clean baseline)
- Final Random Forest modelini
- Lag'li LightGBM modelini
AYNI TEST SETİ üzerinde karşılaştırmak
"""

import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, classification_report
import joblib

# ======================================================
# PATHS
# ======================================================
BASE_DIR = Path(__file__).resolve().parent.parent
TEST_PATH = BASE_DIR / "data" / "test" / "btc_test.csv"
MODEL_DIR = BASE_DIR / "models"

# ======================================================
# LOAD TEST DATA
# ======================================================
df = pd.read_csv(TEST_PATH)

# ======================================================
# LAG FEATURE ENGINEERING (TEST SET)
# Train'de ne yapıldıysa test'te de AYNI
# ======================================================
if "return" in df.columns and "volatility" in df.columns:
    df["return_lag1"] = df["return"].shift(1)
    df["volatility_lag1"] = df["volatility"].shift(1)

    df = df.dropna().reset_index(drop=True)

y_test = df["risk"]

results = []

# ======================================================
# LOGISTIC REGRESSION (CLEAN BASELINE)
# ======================================================
log_model = joblib.load(MODEL_DIR / "logistic_regression_clean.pkl")
log_features = log_model.feature_names_in_

X_log = df[log_features]
y_pred_log = log_model.predict(X_log)

results.append({
    "Model": "Logistic Regression (Clean Baseline)",
    "Accuracy": round(accuracy_score(y_test, y_pred_log), 4),
    "F1-score": round(f1_score(y_test, y_pred_log, average="weighted"), 4)
})

print("\nLOGISTIC REGRESSION (CLEAN)")
print("=" * 60)
print(classification_report(y_test, y_pred_log))

# ======================================================
# RANDOM FOREST (FINAL)
# ======================================================
rf_model = joblib.load(MODEL_DIR / "rf_risk_final_v1.pkl")
rf_features = rf_model.feature_names_in_

X_rf = df[rf_features]
y_pred_rf = rf_model.predict(X_rf)

results.append({
    "Model": "Random Forest (Final)",
    "Accuracy": round(accuracy_score(y_test, y_pred_rf), 4),
    "F1-score": round(f1_score(y_test, y_pred_rf, average="weighted"), 4)
})

print("\nRANDOM FOREST (FINAL)")
print("=" * 60)
print(classification_report(y_test, y_pred_rf))

# ======================================================
# LIGHTGBM (LAG'LI - CLEAN FINAL)
# ======================================================
lgbm_model = joblib.load(MODEL_DIR / "lightgbm_model_lagged.pkl")
lgbm_features = lgbm_model.feature_names_in_

X_lgbm = df[lgbm_features]
y_pred_lgbm = lgbm_model.predict(X_lgbm)

results.append({
    "Model": "LightGBM (Lagged - Clean Final)",
    "Accuracy": round(accuracy_score(y_test, y_pred_lgbm), 4),
    "F1-score": round(f1_score(y_test, y_pred_lgbm, average="weighted"), 4)
})

print("\nLIGHTGBM (LAGGED - CLEAN FINAL)")
print("=" * 60)
print(classification_report(y_test, y_pred_lgbm))

# ======================================================
# FINAL COMPARISON TABLE
# ======================================================
results_df = pd.DataFrame(results)

print("\nFINAL CLEAN COMPARISON (TEST SET)")
print(results_df)
