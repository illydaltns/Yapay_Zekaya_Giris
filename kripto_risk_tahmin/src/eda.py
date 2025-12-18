import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
from pathlib import Path

# =========================
# Proje kök dizini
# =========================
BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_PATH = BASE_DIR / "data" / "train" / "btc_train.csv"
TEST_PATH  = BASE_DIR / "data" / "test"  / "btc_test.csv"

train_df = pd.read_csv(TRAIN_PATH, parse_dates=["date"])
test_df  = pd.read_csv(TEST_PATH,  parse_dates=["date"])

eda_df = (
    pd.concat([train_df, test_df], axis=0)
      .sort_values("date")
      .reset_index(drop=True)
)

print("EDA veri boyutu:", eda_df.shape)
print(eda_df.info())
print(eda_df.head())

print("\nİstatistiksel Özet:")
print(eda_df.describe())

plt.figure(figsize=(6,4))
sns.countplot(x="risk", data=eda_df)
plt.title("Risk Sınıf Dağılımı")
plt.xlabel("Risk (0=Low, 1=Medium, 2=High)")
plt.ylabel("Gözlem Sayısı")
plt.show()

ohlc_cols = ["open", "high", "low", "close", "volume"]

for col in ohlc_cols:
    plt.figure(figsize=(6,4))
    sns.histplot(eda_df[col], bins=50, kde=True)
    plt.title(col.upper())
    plt.show()

derived_cols = ["return", "volatility", "range", "body"]

for col in derived_cols:
    plt.figure(figsize=(6,4))
    sns.histplot(eda_df[col], bins=50, kde=True)
    plt.title(col)
    plt.show()

for col in ohlc_cols:
    plt.figure(figsize=(6,4))
    sns.boxplot(x="risk", y=col, data=eda_df)
    plt.title(col.upper())
    plt.show()

for col in derived_cols:
    plt.figure(figsize=(6,4))
    sns.boxplot(x="risk", y=col, data=eda_df)
    plt.title(col)
    plt.show()

corr_cols = [
    "open", "high", "low", "close", "volume",
    "return", "volatility", "range", "body", "risk"
]

plt.figure(figsize=(10,8))
corr = eda_df[corr_cols].corr()
sns.heatmap(corr, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

plt.figure(figsize=(12,4))
plt.plot(eda_df["date"], eda_df["close"])
plt.title("BTC Close Price")
plt.xlabel("Date")
plt.ylabel("Close")
plt.show()

plt.figure(figsize=(12,4))
plt.plot(eda_df["date"], eda_df["volatility"])
plt.title("BTC Volatility")
plt.xlabel("Date")
plt.ylabel("Volatility")
plt.show()

plt.figure(figsize=(6,4))
sns.scatterplot(
    data=eda_df,
    x="return",
    y="volatility",
    hue="risk",
    alpha=0.6
)
plt.title("Return vs Volatility")
plt.show()

print("Eksik veri kontrolü:")
print(eda_df.isnull().sum())

msno.bar(eda_df)
plt.show()

msno.matrix(eda_df)
plt.show()
