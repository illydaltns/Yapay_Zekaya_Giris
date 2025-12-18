import pandas as pd
from pathlib import Path

# Proje kök dizini
BASE_DIR = Path(__file__).resolve().parent.parent

# Dosya yolları
RAW_PATH = BASE_DIR / "data" / "raw" / "BTC_2019_2023_1d.csv"
TRAIN_PATH = BASE_DIR / "data" / "train" / "btc_train.csv"
TEST_PATH = BASE_DIR / "data" / "test" / "btc_test.csv"

# CSV'yi oku
# (ilk boş index sütununu atlıyoruz)
df = pd.read_csv(RAW_PATH, index_col=0)

# Tarih sütununu standartlaştır
df = df.rename(columns={"datetime": "date"})
df["date"] = pd.to_datetime(df["date"])

# Zaman bazlı train / test ayırımı
train_df = df[df["date"] < "2023-01-01"]
test_df  = df[df["date"] >= "2023-01-01"]

# Kaydet
train_df.to_csv(TRAIN_PATH, index=False)
test_df.to_csv(TEST_PATH, index=False)

print("Split tamamlandı.")
print(f"Train satır sayısı: {len(train_df)}")
print(f"Test satır sayısı : {len(test_df)}")
