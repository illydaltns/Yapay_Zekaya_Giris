import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# =========================
# Yollar
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
# Feature / Target
# =========================
FEATURES = [
    "return",
    "volatility",
    "range",
    "body",
    "volume"
]

TARGET = "risk"

X = df[FEATURES]
y = df[TARGET]

# =========================
# Linear Regression
# =========================
model = LinearRegression()
model.fit(X, y)

# =========================
# Train tahmin
# =========================
y_pred = model.predict(X)

# =========================
# Performans
# =========================
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

print("Train MSE:", mse)
print("Train R2:", r2)

# =========================
# Modeli kaydet
# =========================
joblib.dump(model, MODEL_PATH / "linear_regression.pkl")
