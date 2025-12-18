import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# =========================
# Proje dizinleri
# =========================
BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_PATH = BASE_DIR / "data" / "train" / "btc_train.csv"
MODEL_PATH = BASE_DIR / "models"

MODEL_PATH.mkdir(exist_ok=True)

# =========================
# Veriyi oku
# =========================
df = pd.read_csv(TRAIN_PATH)

# =========================
# Feature / Label ayrımı
# =========================
FEATURES = [
    "return",
    "range",
    "body",
    "volume"
]

TARGET = "risk"

X = df[FEATURES]
y = df[TARGET]

# =========================
# Train / Validation split
# (zaman bozulmasın diye shuffle=False)
# =========================
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    shuffle=False
)

# =========================
# Model tanımı
# =========================
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=6,
    random_state=42,
    class_weight="balanced"
)

# =========================
# Model eğitimi
# =========================
model.fit(X_train, y_train)

# =========================
# Basit validation kontrolü
# =========================
y_pred = model.predict(X_val)
acc = accuracy_score(y_val, y_pred)

print("Model eğitildi.")
print(f"Validation Accuracy: {acc:.4f}")

# =========================
# Modeli kaydet
# =========================
joblib.dump(model, MODEL_PATH / "risk_model_no_vol.pkl")
print("Model kaydedildi: models/risk_model_no_vol.pkl")
