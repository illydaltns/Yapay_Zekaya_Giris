import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
from pathlib import Path

# =========================
# Görselleştirme Ayarları
# =========================
sns.set_theme(style="whitegrid", context="notebook", palette="muted")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.labelsize"] = 12

# =========================
# Proje kök dizini
# =========================
BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_PATH = BASE_DIR / "data" / "train" / "btc_train.csv"
TEST_PATH  = BASE_DIR / "data" / "test"  / "btc_test.csv"


def plot_risk_distribution(df):
    """Risk sınıf dağılımını çizer."""
    plt.figure(figsize=(8, 6))
    ax = sns.countplot(x="risk", data=df, palette="viridis", hue="risk", legend=False)
    plt.title("Risk Sınıf Dağılımı", fontsize=16, fontweight="bold")
    plt.xlabel("Risk (0=Low, 1=Medium, 2=High)")
    plt.ylabel("Gözlem Sayısı")
    
    # Barların üzerine sayıları ekle
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', xytext=(0, 10), textcoords='offset points')
    plt.show()

def plot_distributions(df, cols, title_prefix=""):
    """Sütunların histogram dağılımlarını çizer."""
    for col in cols:
        plt.figure(figsize=(10, 6))
        sns.histplot(df[col], bins=50, kde=True, color="steelblue", edgecolor="black")
        plt.title(f"{title_prefix}{col.upper()} Dağılımı", fontsize=14)
        plt.xlabel(col.upper())
        plt.ylabel("Frekans")
        plt.show()

def plot_boxplots(df, cols):
    """Risk sınıflarına göre boxplot çizer."""
    for col in cols:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x="risk", y=col, data=df, palette="coolwarm")
        plt.title(f"Risk Gruplarına Göre {col.upper()}", fontsize=14)
        plt.show()

def plot_correlation_heatmap(df, cols):
    """Korelasyon matrisini çizer."""
    plt.figure(figsize=(12, 10))
    corr = df[cols].corr()
    
    mask = None # İsterseniz üçgen maske ekleyebilirsiniz: np.triu(np.ones_like(corr, dtype=bool))
    
    sns.heatmap(
        corr, 
        annot=True, 
        fmt=".2f", 
        cmap="coolwarm", 
        center=0, 
        linewidths=0.5, 
        square=True,
        cbar_kws={"shrink": .8}
    )
    plt.title("Değişkenler Arası Korelasyon Matrisi", fontsize=16, fontweight="bold")
    plt.show()

def plot_time_series(df, col, title):
    """Zaman serisi grafiği çizer."""
    plt.figure(figsize=(14, 6))
    plt.plot(df["date"], df[col], label=col, linewidth=1.5, color="darkblue")
    plt.title(title, fontsize=16)
    plt.xlabel("Tarih")
    plt.ylabel(col.capitalize())
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.show()

def plot_scatter_volatility_return(df):
    """Return vs Volatility scatter plot."""
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=df,
        x="return",
        y="volatility",
        hue="risk",
        palette="deep",
        alpha=0.7,
        s=60
    )
    plt.title("Return vs Volatility (Risk Gruplu)", fontsize=16)
    plt.axhline(0, color='grey', linestyle='--', linewidth=1)
    plt.axvline(0, color='grey', linestyle='--', linewidth=1)
    plt.show()


# =========================
# Veri Yükleme ve Hazırlık
# =========================
print("Veriler yükleniyor...")
train_df = pd.read_csv(TRAIN_PATH, parse_dates=["date"])
test_df  = pd.read_csv(TEST_PATH,  parse_dates=["date"])

eda_df = (
    pd.concat([train_df, test_df], axis=0)
      .sort_values("date")
      .reset_index(drop=True)
)

print("EDA veri boyutu:", eda_df.shape)
print(eda_df.info())
print("\nİstatistiksel Özet:")
print(eda_df.describe().T)

# =========================
# Görselleştirmeler
# =========================

# 1. Risk Dağılımı
plot_risk_distribution(eda_df)

ohlc_cols = ["open", "high", "low", "close", "volume"]
derived_cols = ["return", "volatility", "range", "body"]

# 2. Dağılımlar
plot_distributions(eda_df, ohlc_cols, "OHLC - ")
plot_distributions(eda_df, derived_cols, "Türetilmiş - ")

# 3. Kutu Grafikleri (Outlier Analizi & Risk İlişkisi)
plot_boxplots(eda_df, ohlc_cols + derived_cols)

# 4. Korelasyon Matrisi
corr_cols = ohlc_cols + derived_cols + ["risk"]
plot_correlation_heatmap(eda_df, corr_cols)

# 5. Zaman Serileri
plot_time_series(eda_df, "close", "Bitcoin Kapanış Fiyatı Zaman Serisi")
plot_time_series(eda_df, "volatility", "Bitcoin Volatilite Zaman Serisi")

# 6. Scatter Plot
plot_scatter_volatility_return(eda_df)

# 7. Eksik Veri Analizi (Matrix Grafiği)
print("\nEksik veri kontrolü:")
print(eda_df.isnull().sum())

# Missingno Matrix Grafiği
msno.matrix(eda_df, sparkline=False, figsize=(10, 6), fontsize=12, color=(0.2, 0.4, 0.6))
plt.title("Eksik Veri Matrisi", fontsize=16)
plt.show()