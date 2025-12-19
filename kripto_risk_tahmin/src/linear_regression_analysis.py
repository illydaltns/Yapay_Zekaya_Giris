import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# =========================
# Yollar
# =========================
BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN_PATH = BASE_DIR / "data" / "train" / "btc_train.csv"

# =========================
# Veriyi oku
# =========================
df = pd.read_csv(TRAIN_PATH)

# =========================
# FEATURE / TARGET
# =========================
FEATURES = [
    "return",
    "range",
    "body",
    "volume",
    "range_pct",
    "return_lag_1",
    "return_lag_3",
    "ma_7_diff"
]

TARGET = "volatility"   # ✅ DOĞRU HEDEF

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
