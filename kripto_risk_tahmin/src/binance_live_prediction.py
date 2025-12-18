"""
binance_live_prediction.py

Binance API üzerinden son günlük BTC verisini çekip
eğitilmiş final modeli ile risk tahmini yapar.
"""

import pandas as pd
from pathlib import Path
from binance.client import Client
import joblib

# ======================================================
# BINANCE (PUBLIC API - KEY GEREKMİYOR)
# ======================================================
client = Client()

SYMBOL = "BTCUSDT"
INTERVAL = Client.KLINE_INTERVAL_1DAY
LOOKBACK_DAYS = 60  # feature'lar için yeterli pencere

# ======================================================
# PROJE DİZİNLERİ
# ======================================================
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "rf_risk_final_v1.pkl"

# ======================================================
# FEATURE SET (FINAL MODEL)
# ======================================================
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

# Sayısal kolonlar
for col in ["open", "high", "low", "close", "volume"]:
    df[col] = df[col].astype(float)

df = df.sort_values("date")

# ======================================================
# 2️⃣ FEATURE ENGINEERING (AYNI MANTIK)
# ======================================================
df["return"] = df["close"].pct_change()
df["range"] = df["high"] - df["low"]
df["body"] = df["close"] - df["open"]

df["return_lag_1"] = df["return"].shift(1)
df["return_lag_3"] = df["return"].shift(3)

df["ma_7"] = df["close"].rolling(7).mean()
df["ma_7_diff"] = (df["close"] - df["ma_7"]) / df["ma_7"]

df["range_pct"] = df["range"] / df["close"]

df = df.dropna()

# ======================================================
# 3️⃣ MODELİ YÜKLE
# ======================================================
model = joblib.load(MODEL_PATH)

# ======================================================
# 4️⃣ TAHMİN (SON GÜN)
# ======================================================
latest = df.iloc[-1]
X_latest = latest[FEATURES].values.reshape(1, -1)

risk_pred = model.predict(X_latest)[0]

risk_map = {
    0: "LOW RISK 🟢",
    1: "MEDIUM RISK 🟡",
    2: "HIGH RISK 🔴"
}

print("=" * 60)
print("BINANCE LIVE RISK PREDICTION")
print("=" * 60)
print(f"Date : {latest['date'].date()}")
print(f"Price: {latest['close']:.2f} USDT")
print(f"Risk : {risk_map[risk_pred]}")
print("=" * 60)
