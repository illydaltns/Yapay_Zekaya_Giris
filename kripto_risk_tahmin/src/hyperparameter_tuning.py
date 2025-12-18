"""
hyperparameter_tuning.py

Random Forest için mini hyperparameter denemesi.
Amaç: Proxy model (volatility olmadan) için en uygun parametreleri bulmak.
"""

import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import itertools

# ======================================================
# PROJE DİZİNLERİ
# ======================================================
BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN_PATH = BASE_DIR / "data" / "train" / "btc_train.csv"

# ======================================================
# FEATURE SET (VOLATILITY YOK)
# ======================================================
FEATURES = ["close", "volume", "return"]
TARGET = "risk"

# ======================================================
# VERİYİ OKU
# ======================================================
df = pd.read_csv(TRAIN_PATH)

X = df[FEATURES]
y = df[TARGET]

X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    shuffle=False
)

# ======================================================
# PARAMETRE ARALIKLARI (KÜÇÜK VE ANLAMLI)
# ======================================================
param_grid = {
    "n_estimators": [100, 300],
    "max_depth": [6, 10],
    "min_samples_leaf": [3, 7]
}

# ======================================================
# DENEMELER
# ======================================================
results = []

for n_estimators, max_depth, min_samples_leaf in itertools.product(
    param_grid["n_estimators"],
    param_grid["max_depth"],
    param_grid["min_samples_leaf"]
):
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)

    results.append({
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "min_samples_leaf": min_samples_leaf,
        "accuracy": acc
    })

# ======================================================
# SONUÇLARI GÖSTER
# ======================================================
results_df = pd.DataFrame(results).sort_values(
    by="accuracy", ascending=False
)

print("=" * 70)
print("HYPERPARAMETER TUNING RESULTS (Proxy Model)")
print("=" * 70)
print(results_df)
