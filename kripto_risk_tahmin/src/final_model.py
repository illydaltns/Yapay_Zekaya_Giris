"""
final_model.py

Final Random Forest modelinin eğitimi.
Enhanced feature set + en iyi hiperparametreler kullanılır.
"""

import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# ======================================================
# DİZİNLER
# ======================================================
BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_PATH = BASE_DIR / "data" / "train" / "btc_train.csv"
TEST_PATH  = BASE_DIR / "data" / "test"  / "btc_test.csv"
MODEL_DIR  = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "rf_risk_final_v1.pkl"

# ======================================================
# FEATURE SET (ENHANCED)
# ======================================================
FEATURES = [
    "close",
    "volume",
    "return",
    "body",
    "range",
    "return_lag_1",
    "return_lag_3",
    "ma_7_diff",
    "range_pct"
]

TARGET = "risk"

# ======================================================
# VERİYİ OKU
# ======================================================
train_df = pd.read_csv(TRAIN_PATH)
test_df  = pd.read_csv(TEST_PATH)

X_train = train_df[FEATURES]
y_train = train_df[TARGET]

X_test = test_df[FEATURES]
y_test = test_df[TARGET]

# ======================================================
# FINAL MODEL (BEST PARAMS)
# ======================================================
model = RandomForestClassifier(
    n_estimators=400,
    max_depth=6,
    min_samples_leaf=3,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

# ======================================================
# TRAIN
# ======================================================
print("=" * 70)
print("FINAL MODEL TRAINING STARTED")
print("=" * 70)

model.fit(X_train, y_train)

# ======================================================
# EVALUATION - TRAIN
# ======================================================
train_pred = model.predict(X_train)
train_acc = accuracy_score(y_train, train_pred)

print("\nTRAIN PERFORMANCE")
print("-" * 40)
print(f"Train Accuracy: {train_acc:.4f}")

# ======================================================
# EVALUATION - TEST
# ======================================================
test_pred = model.predict(X_test)
test_acc = accuracy_score(y_test, test_pred)

print("\nTEST PERFORMANCE")
print("-" * 40)
print(f"Test Accuracy: {test_acc:.4f}\n")

print("Classification Report (Test):")
print(classification_report(y_test, test_pred))

print("Confusion Matrix (Test):")
print(confusion_matrix(y_test, test_pred))

# ======================================================
# SAVE MODEL
# ======================================================
joblib.dump(model, MODEL_PATH)

print("\n" + "=" * 70)
print(f"FINAL MODEL SAVED → {MODEL_PATH}")
print("=" * 70)
