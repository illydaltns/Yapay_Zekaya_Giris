"""
09_train_model.py

Bu dosyada, feature selection aşamasında belirlenen nihai feature seti
kullanılarak Random Forest modeli eğitilmektedir.

Bu aşamada:
- Feature seçimi yapılmaz
- Veri sızıntısı (leakage) yoktur
- Model zaman sırası korunarak eğitilir
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
# Feature Selection Aşamasında Belirlenen Değişkenler
# ======================================================
FEATURES = [
    "close",
    "volume",
    "return",
    "volatility"
]

TARGET = "risk"

# ======================================================
# Veriyi Oku
# ======================================================
df = pd.read_csv(TRAIN_PATH)

X = df[FEATURES]
y = df[TARGET]

# ======================================================
# Train / Validation Split
# (Zaman serisi olduğu için shuffle=False)
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
    max_depth=8,
    min_samples_split=10,
    min_samples_leaf=5,
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
print("MODEL EĞİTİMİ TAMAMLANDI")
print("=" * 60)

print(f"\nValidation Accuracy: {acc:.4f}\n")

print("Classification Report:")
print(classification_report(y_val, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_val, y_pred))

# ======================================================
# Modeli Kaydet
# ======================================================
model_path = MODEL_DIR / "risk_model_random_forest.pkl"
joblib.dump(model, model_path)

print(f"\nModel kaydedildi: {model_path}")
