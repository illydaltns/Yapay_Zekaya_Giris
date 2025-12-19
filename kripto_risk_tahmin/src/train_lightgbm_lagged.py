# ======================================================
# train_lightgbm_lagged.py
# Kripto Risk Tahmini - LightGBM (Lag'li Feature'lar)
# ======================================================

import pandas as pd
from pathlib import Path
import joblib

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, f1_score

from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt


# ======================================================
# 1. PROJE DİZİNLERİ
# ======================================================
BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_PATH = BASE_DIR / "data" / "train" / "btc_train.csv"
MODEL_DIR  = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "lightgbm_model_lagged.pkl"


# ======================================================
# 2. VERİYİ OKU
# ======================================================
df = pd.read_csv(TRAIN_PATH)

print("Veri seti boyutu (ilk):", df.shape)


# ======================================================
# 3. LAG'Lİ FEATURE ENGINEERING
# ======================================================
# Gelecek bilgisi sızıntısını (leakage) önlemek için
df["return_lag1"] = df["return"].shift(1)
df["volatility_lag1"] = df["volatility"].shift(1)

# Lag sonrası NaN oluşur → temizle
df = df.dropna().reset_index(drop=True)

print("Veri seti boyutu (lag sonrası):", df.shape)


# ======================================================
# 4. FEATURE / TARGET
# ======================================================
FEATURES = [
    "return_lag1",
    "volatility_lag1",
    "range",
    "body",
    "volume"
]

TARGET = "risk"

X = df[FEATURES]
y = df[TARGET]


# ======================================================
# 5. LIGHTGBM MODELİ
# ======================================================
model = LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=31,
    max_depth=-1,
    random_state=42
)


# ======================================================
# 6. TIME SERIES SPLIT
# ======================================================
tscv = TimeSeriesSplit(n_splits=5)

acc_scores = []
f1_scores = []


# ======================================================
# 7. EĞİTİM & VALIDATION
# ======================================================
for fold, (train_idx, val_idx) in enumerate(tscv.split(X), start=1):
    print(f"\n--- Fold {fold} ---")

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)

    acc = accuracy_score(y_val, y_pred)
    f1  = f1_score(y_val, y_pred, average="weighted")

    acc_scores.append(acc)
    f1_scores.append(f1)

    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")


# ======================================================
# 8. ORTALAMA SONUÇLAR
# ======================================================
mean_acc = sum(acc_scores) / len(acc_scores)
mean_f1  = sum(f1_scores) / len(f1_scores)

print("\n===================================")
print("LIGHTGBM (LAG'Lİ) ORTALAMA PERFORMANS")
print("===================================")
print(f"Accuracy: {mean_acc:.4f}")
print(f"F1 Score: {mean_f1:.4f}")


# ======================================================
# 9. MODELİ KAYDET
# ======================================================
joblib.dump(model, MODEL_PATH)
print(f"\nModel kaydedildi -> {MODEL_PATH}")


# ======================================================
# 10. FEATURE IMPORTANCE
# ======================================================
feature_importance = pd.DataFrame({
    "feature": FEATURES,
    "importance": model.feature_importances_
}).sort_values(by="importance", ascending=False)

print("\nFeature Importance (Lag'li):")
print(feature_importance)

plt.figure(figsize=(8, 4))
plt.barh(feature_importance["feature"], feature_importance["importance"])
plt.gca().invert_yaxis()
plt.title("LightGBM Feature Importance (Lag'li)")
plt.tight_layout()
plt.show()
