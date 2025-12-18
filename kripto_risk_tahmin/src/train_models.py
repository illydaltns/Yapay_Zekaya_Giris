"""
train_models.py

Bu dosyada farklı feature stratejileri kullanılarak
Random Forest modelleri eğitilir ve standart bir isimlendirme
ile kaydedilir.

Şu an eğitilen modeller:
1) Random Forest - Baseline (with volatility)
2) Random Forest - Proxy model (basic features)
3) Random Forest - Proxy model (enhanced features)

Yeni modeller bu yapı bozulmadan eklenebilir.
"""

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# ======================================================
# PROJE DİZİNLERİ
# ======================================================
BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_PATH = BASE_DIR / "data" / "train" / "btc_train.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

TARGET = "risk"

# ======================================================
# MODEL KONFİGÜRASYONLARI
# ======================================================
MODEL_CONFIGS = [
    {
        "name": "rf_risk_baseline_with_volatility_v1",
        "features": ["close", "volume", "return", "volatility"],
        "description": "Baseline model (upper bound, volatility included)"
    },
    {
        "name": "rf_risk_proxy_basic_v1",
        "features": ["close", "volume", "return"],
        "description": "Proxy model (basic market features)"
    },
    {
        "name": "rf_risk_proxy_enhanced_v1",
        "features": [
            "close",
            "volume",
            "return",
            "body",
            "range",
            "return_lag_1",
            "return_lag_3",
            "ma_7_diff",
            "range_pct"
        ],
        "description": "Proxy model (enhanced engineered features)"
    }
]

# ======================================================
# VERİYİ OKU
# ======================================================
df = pd.read_csv(TRAIN_PATH)

# ======================================================
# MODEL EĞİTİM DÖNGÜSÜ
# ======================================================
for cfg in MODEL_CONFIGS:
    print("\n" + "=" * 75)
    print(f"TRAINING MODEL : {cfg['name']}")
    print(f"DESCRIPTION    : {cfg['description']}")
    print(f"FEATURE COUNT  : {len(cfg['features'])}")
    print("=" * 75)

    X = df[cfg["features"]]
    y = df[TARGET]

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        shuffle=False
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_split=15,
        min_samples_leaf=7,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )

    # ----------------------
    # TRAIN
    # ----------------------
    model.fit(X_train, y_train)

    # ----------------------
    # VALIDATION
    # ----------------------
    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)

    print(f"\nValidation Accuracy: {acc:.4f}\n")
    print("Classification Report:")
    print(classification_report(y_val, y_pred))

    # ----------------------
    # SAVE MODEL
    # ----------------------
    model_path = MODEL_DIR / f"{cfg['name']}.pkl"
    joblib.dump(model, model_path)

    print(f"\nModel saved → {model_path}")

print("\nALL MODELS TRAINED SUCCESSFULLY.")
