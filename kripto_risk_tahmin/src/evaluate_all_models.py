"""
evaluate_final_models.py

Amaç:
- Logistic Regression (baseline) ve
- Random Forest (final) modellerini
AYNI TEST SETİ üzerinde karşılaştırmak.

Her model kendi feature setiyle değerlendirilir.
"""

import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent
TEST_PATH = BASE_DIR / "data" / "test" / "btc_test.csv"
MODEL_DIR = BASE_DIR / "models"

df = pd.read_csv(TEST_PATH)
y_test = df["risk"]

results = []

# ======================================================
# LOGISTIC REGRESSION (BASELINE)
# ======================================================
log_model = joblib.load(MODEL_DIR / "logistic_regression_baseline.pkl")
log_features = log_model.feature_names_in_

X_log = df[log_features]
y_pred_log = log_model.predict(X_log)

results.append({
    "Model": "Logistic Regression (Baseline)",
    "Accuracy": round(accuracy_score(y_test, y_pred_log), 4),
    "F1-score": round(f1_score(y_test, y_pred_log, average="weighted"), 4)
})

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

# ======================================================
# RESULTS
# ======================================================
results_df = pd.DataFrame(results)
print("\nFINAL TEST SET COMPARISON")
print(results_df)
