"""
train_logistic_regression.py

Amaç:
- Bitcoin risk tahmini için Logistic Regression baseline modeli
- Random Forest ile adil karşılaştırma yapmak
"""

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

# ======================================================
# PATHS
# ======================================================
BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN_PATH = BASE_DIR / "data" / "train" / "btc_train.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

# ======================================================
# LOAD DATA
# ======================================================
df = pd.read_csv(TRAIN_PATH)

TARGET = "risk"

# ======================================================
# FEATURE SET
# (Feature engineering çıktısıyla UYUMLU)
# ======================================================
FEATURES = [
    "return",
    "volatility",
    "range",
    "body",
    "range_pct",
    "return_lag_1",
    "return_lag_3",
    "ma_7_diff"
]

X = df[FEATURES]
y = df[TARGET]

# ======================================================
# TRAIN / VALIDATION SPLIT
# (Time series olduğu için shuffle=False)
# ======================================================
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    shuffle=False
)

# ======================================================
# PIPELINE (Scaling + Logistic Regression)
# ======================================================
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="lbfgs"
    ))
])

# ======================================================
# TRAIN
# ======================================================
pipeline.fit(X_train, y_train)

# ======================================================
# VALIDATION
# ======================================================
y_pred = pipeline.predict(X_val)

acc = accuracy_score(y_val, y_pred)

print("\n" + "=" * 60)
print("LOGISTIC REGRESSION (BASELINE) RESULTS")
print("=" * 60)
print(f"Validation Accuracy: {acc:.4f}\n")
print("Classification Report:")
print(classification_report(y_val, y_pred))

# ======================================================
# SAVE MODEL
# ======================================================
model_path = MODEL_DIR / "logistic_regression_baseline.pkl"
joblib.dump(pipeline, model_path)

print(f"\nModel saved → {model_path}")
