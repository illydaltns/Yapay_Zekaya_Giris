"""
feature_importance_analysis.py

Random Forest modelleri için feature importance analizi.
Yeni isimlendirme standardına uygundur.
"""

import joblib
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ======================================================
# PROJE DİZİNLERİ
# ======================================================
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

# ======================================================
# MODEL VE FEATURE TANIMLARI
# ======================================================
MODELS = {
    "Baseline (With Volatility)": {
        "path": MODEL_DIR / "random_forest_risk_with_volatility_v1.pkl",
        "features": ["close", "volume", "return", "volatility"]
    },
    "Proxy Model (Without Volatility)": {
        "path": MODEL_DIR / "random_forest_risk_proxy_features_v1.pkl",
        "features": ["close", "volume", "return"]
    }
}

# ======================================================
# FEATURE IMPORTANCE ANALİZİ
# ======================================================
all_results = []

for model_name, config in MODELS.items():
    print(f"\nLoading model: {model_name}")
    model = joblib.load(config["path"])

    importances = model.feature_importances_

    df = pd.DataFrame({
        "model": model_name,
        "feature": config["features"],
        "importance": importances
    })

    all_results.append(df)

fi_df = pd.concat(all_results).reset_index(drop=True)

print("\n" + "=" * 70)
print("FEATURE IMPORTANCE ANALYSIS")
print("=" * 70)
print(fi_df)

# ======================================================
# GÖRSELLEŞTİRME
# ======================================================
for model_name in fi_df["model"].unique():
    subset = fi_df[fi_df["model"] == model_name] \
        .sort_values(by="importance", ascending=False)

    plt.figure(figsize=(8, 5))
    plt.barh(subset["feature"], subset["importance"])
    plt.title(f"Feature Importance - {model_name}")
    plt.xlabel("Importance")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()
