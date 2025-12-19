"""
train_logistic_regression_clean.py

Amaç:
- Volatility içermeyen
- Daha adil Logistic Regression baseline modeli
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
# CLEAN FEATURE SET (NO VOLATILITY)
# ======================================================
FEATURES = [
    "return",
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
# ======================================================
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    shuffle=False
)

# ======================================================
# PIPELINE
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

print("\nLOGISTIC REGRESSION (CLEAN BASELINE)")
print("=" * 60)
print(f"Validation Accuracy: {accuracy_score(y_val, y_pred):.4f}")
print("Classification Report:")
print(classification_report(y_val, y_pred))

# ======================================================
# SAVE
# ======================================================
model_path = MODEL_DIR / "logistic_regression_clean.pkl"
joblib.dump(pipeline, model_path)

print(f"\nModel saved → {model_path}")
