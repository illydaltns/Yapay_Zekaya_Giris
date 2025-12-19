"""
binance_live_prediction.py

Binance API üzerinden güncel BTC verisini çekip
LightGBM (Lagged - Clean Final) modeli ile risk tahmini yapar.
"""

import pandas as pd
from pathlib import Path
from binance.client import Client
import joblib

# ======================================================
# BINANCE (PUBLIC API)
# ======================================================
client = Client()

SYMBOL = "BTCUSDT"
INTERVAL = Client.KLINE_INTERVAL_1DAY
LOOKBACK_DAYS = 60  # volatility + lag için yeterli

# ======================================================
# PROJE DİZİNLERİ
# ======================================================
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "lightgbm_model_lagged.pkl"

# ======================================================
# MODELİ YÜKLE
# ======================================================
model = joblib.load(MODEL_PATH)
FEATURES = model.feature_names_in_

# ======================================================
# 1️⃣ VERİYİ ÇEK
# ======================================================
klines = client.get_klines(
    symbol=SYMBOL,
    interval=INTERVAL,
    limit=LOOKBACK_DAYS
)

df = pd.DataFrame(klines, columns=[
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_asset_volume",
    "number_of_trades", "taker_buy_base",
    "taker_buy_quote", "ignore"
])

df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
df = df.rename(columns={"open_time": "date"})

for col in ["open", "high", "low", "close", "volume"]:
    df[col] = df[col].astype(float)

df = df.sort_values("date")

# ======================================================
# 2️⃣ FEATURE ENGINEERING (TRAIN İLE AYNI)
# ======================================================
df["return"] = df["close"].pct_change()
df["volatility"] = df["return"].rolling(14).std()
df["range"] = df["high"] - df["low"]
df["body"] = df["close"] - df["open"]

# LAG
df["return_lag1"] = df["return"].shift(1)
df["volatility_lag1"] = df["volatility"].shift(1)

df = df.dropna()

# ======================================================
# 3️⃣ TAHMİN (SON GÜN)
# ======================================================
X_live = df[FEATURES].iloc[-1:].copy()
risk_pred = model.predict(X_live)[0]

risk_map = {
    0: "LOW RISK 🟢",
    1: "MEDIUM RISK 🟡",
    2: "HIGH RISK 🔴"
}

latest = df.iloc[-1]

print("=" * 60)
print("BINANCE LIVE RISK PREDICTION (LIGHTGBM)")
print("=" * 60)
print(f"Date : {latest['date'].date()}")
print(f"Price: {latest['close']:.2f} USDT")
print(f"Risk : {risk_map[risk_pred]}")
print("=" * 60)
