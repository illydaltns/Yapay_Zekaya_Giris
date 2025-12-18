"""
10_train_model_no_volatility.py

Bu model, risk etiketinin üretiminde kullanılan volatilite değişkeni
MODELE DAHİL EDİLMEDEN eğitilmektedir.

Amaç:
- Modelin volatiliteyi dolaylı feature'lar üzerinden öğrenip öğrenemediğini test etmek
- Gerçek tahmin gücünü ölçmek
"""

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# ======================================================
# Proje Dizinleri
# ======================================================
BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_PATH = BASE_DIR / "data" / "train" / "btc_train.csv"
MODEL_DIR  = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

# ======================================================
# VOLATILITY OLMADAN KULLANILACAK FEATURE'LAR
# ======================================================
FEATURES = [
    "close",
    "volume",
    "return"
]

TARGET = "risk"

# ======================================================
# Veriyi Oku
# ======================================================
df = pd.read_csv(TRAIN_PATH)

X = df[FEATURES]
y = df[TARGET]

# ======================================================
# Train / Validation Split (Zaman sırası korunur)
# ======================================================
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    shuffle=False
)

# ======================================================
# Random Forest Model Tanımı
# ======================================================
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_split=15,
    min_samples_leaf=7,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

# ======================================================
# Model Eğitimi
# ======================================================
model.fit(X_train, y_train)

# ======================================================
# Validation Performansı
# ======================================================
y_pred = model.predict(X_val)

acc = accuracy_score(y_val, y_pred)

print("=" * 60)
print("VOLATILITY OLMADAN MODEL EĞİTİLDİ")
print("=" * 60)

print(f"\nValidation Accuracy: {acc:.4f}\n")

print("Classification Report:")
print(classification_report(y_val, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_val, y_pred))

# ======================================================
# Modeli Kaydet
# ======================================================
model_path = MODEL_DIR / "risk_model_no_volatility.pkl"
joblib.dump(model, model_path)

print(f"\nModel kaydedildi: {model_path}")
