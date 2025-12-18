import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_PATH = BASE_DIR / "data" / "train" / "btc_train.csv"
TEST_PATH  = BASE_DIR / "data" / "test" / "btc_test.csv"

# ======================================================
# Ortak feature fonksiyonu (GENİŞLETİLDİ)
# ======================================================
def add_features(df):
    df = df.sort_values("date").copy()

    # ----------------------
    # ESKİ FEATURE'LAR (AYNEN KORUNDU)
    # ----------------------
    df["return"] = df["close"].pct_change()
    df["volatility"] = df["return"].rolling(window=14).std()
    df["range"] = df["high"] - df["low"]
    df["body"] = df["close"] - df["open"]

    # ----------------------
    # YENİ EKLENEN FEATURE'LAR
    # ----------------------

    # 1️⃣ Gecikmeli getiriler (momentum)
    df["return_lag_1"] = df["return"].shift(1)
    df["return_lag_3"] = df["return"].shift(3)

    # 2️⃣ Hareketli ortalamadan sapma
    df["ma_7"] = df["close"].rolling(7).mean()
    df["ma_7_diff"] = (df["close"] - df["ma_7"]) / df["ma_7"]

    # 3️⃣ Oransal günlük aralık
    df["range_pct"] = df["range"] / df["close"]

    return df.dropna()

# ======================================================
# ===== TRAIN =====
# ======================================================
train_df = pd.read_csv(TRAIN_PATH)
train_df["date"] = pd.to_datetime(train_df["date"])
train_df = add_features(train_df)

# Risk label eşikleri (SADECE train'den)
q1, q2 = train_df["volatility"].quantile([0.33, 0.66])

def risk_label(v):
    if v <= q1:
        return 0   # Low risk
    elif v <= q2:
        return 1   # Medium risk
    else:
        return 2   # High risk

train_df["risk"] = train_df["volatility"].apply(risk_label)

# ======================================================
# ===== TEST =====
# ======================================================
test_df = pd.read_csv(TEST_PATH)
test_df["date"] = pd.to_datetime(test_df["date"])
test_df = add_features(test_df)

# Test için AYNI eşikler
test_df["risk"] = test_df["volatility"].apply(risk_label)

# ======================================================
# KAYDET
# ======================================================
train_df.to_csv(TRAIN_PATH, index=False)
test_df.to_csv(TEST_PATH, index=False)

print("✅ Feature engineering güncellendi (eski kod korunarak).")
print("\nTrain sütunları:")
print(train_df.columns.tolist())
print("\nTrain shape:", train_df.shape)
print("Test shape :", test_df.shape)
