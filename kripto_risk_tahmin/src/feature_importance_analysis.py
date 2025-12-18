"""
feature_importance_analysis.py

Random Forest modeli için feature importance analizi.
Amaç: Proxy enhanced modelin hangi feature'lara dayandığını görmek.
"""

import pandas as pd
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# ======================================================
# DİZİNLER
# ======================================================
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "rf_risk_proxy_enhanced_v1.pkl"
TRAIN_PATH = BASE_DIR / "data" / "train" / "btc_train.csv"

# ======================================================
# MODEL ve VERİ
# ======================================================
model = joblib.load(MODEL_PATH)
df = pd.read_csv(TRAIN_PATH)

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

X = df[FEATURES]

# ======================================================
# FEATURE IMPORTANCE
# ======================================================
importances = model.feature_importances_

importance_df = pd.DataFrame({
    "feature": FEATURES,
    "importance": importances
}).sort_values(by="importance", ascending=False)

print("\nFEATURE IMPORTANCE:")
print(importance_df)

# ======================================================
# GÖRSELLEŞTİRME
# ======================================================
plt.figure(figsize=(10, 6))
sns.barplot(
    x="importance",
    y="feature",
    data=importance_df,
    hue="feature",
    legend=False,
    palette="viridis"
)

plt.title("Feature Importance - Random Forest (Proxy Enhanced)")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()
