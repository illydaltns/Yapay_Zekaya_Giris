import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
from pathlib import Path

# =========================
# Ayarlar
# =========================
# Profesyonel görünüm
sns.set_theme(style="darkgrid", context="talk", palette="viridis")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN_PATH = BASE_DIR / "data" / "train" / "btc_train.csv"
TEST_PATH  = BASE_DIR / "data" / "test"  / "btc_test.csv"
FIGURES_DIR = BASE_DIR / "reports" / "figures"

# Klasör yoksa oluştur
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

def load_data():
    """Verileri yükler ve birleştirir."""
    train_df = pd.read_csv(TRAIN_PATH, parse_dates=["date"])
    test_df  = pd.read_csv(TEST_PATH,  parse_dates=["date"])
    
    df = pd.concat([train_df, test_df], axis=0).sort_values("date").reset_index(drop=True)
    return df

def plot_all_graphics(df):
    """
    Orijinal eda.py içindeki 25 grafiğin hepsini profesyonel stilde üretir ve kaydeder.
    """
    print("Grafikler oluşturuluyor...")

    # 1. Risk Sınıf Dağılımı (1 Grafik)
    plt.figure(figsize=(8, 6))
    ax = sns.countplot(x="risk", data=df, palette="viridis")
    plt.title("1. Risk Sınıf Dağılımı")
    plt.xlabel("Risk (0=Düşük, 1=Orta, 2=Yüksek)")
    plt.ylabel("Gözlem Sayısı")
    # Değerleri yaz
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', fontsize=12, color='black', xytext=(0, 5), textcoords='offset points')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "01_risk_distribution.png")
    plt.close()

    ohlc_cols = ["open", "high", "low", "close", "volume"]
    derived_cols = ["return", "volatility", "range", "body"]

    # 2. OHLC Histogramları (5 Grafik) [2, 3, 4, 5, 6]
    for i, col in enumerate(ohlc_cols, start=2):
        plt.figure(figsize=(8, 5))
        sns.histplot(df[col], bins=50, kde=True, color='teal')
        plt.title(f"{i}. {col.capitalize()} Dağılımı")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"{i:02d}_{col}_hist.png")
        plt.close()

    # 3. Türetilmiş Sütun Histogramları (4 Grafik) [7, 8, 9, 10]
    start_idx = 2 + len(ohlc_cols)
    for i, col in enumerate(derived_cols, start=start_idx):
        plt.figure(figsize=(8, 5))
        sns.histplot(df[col], bins=50, kde=True, color='purple')
        plt.title(f"{i}. {col.capitalize()} Dağılımı")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"{i:02d}_{col}_hist.png")
        plt.close()

    # 4. OHLC Boxplot vs Risk (5 Grafik) [11, 12, 13, 14, 15]
    start_idx += len(derived_cols)
    for i, col in enumerate(ohlc_cols, start=start_idx):
        plt.figure(figsize=(8, 5))
        sns.boxplot(x="risk", y=col, data=df, palette="viridis")
        plt.title(f"{i}. Risk Gruplarına Göre {col.capitalize()}")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"{i:02d}_{col}_boxplot.png")
        plt.close()

    # 5. Türetilmiş Sütun Boxplot vs Risk (4 Grafik) [16, 17, 18, 19]
    start_idx += len(ohlc_cols)
    for i, col in enumerate(derived_cols, start=start_idx):
        plt.figure(figsize=(8, 5))
        sns.boxplot(x="risk", y=col, data=df, palette="viridis")
        plt.title(f"{i}. Risk Gruplarına Göre {col.capitalize()}")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"{i:02d}_{col}_boxplot.png")
        plt.close()
    
    start_idx += len(derived_cols) 

    # 20. Korelasyon Matrisi
    corr_cols = ohlc_cols + derived_cols + ["risk"]
    plt.figure(figsize=(10, 8))
    corr = df[corr_cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", linewidths=.5)
    plt.title(f"{start_idx}. Korelasyon Matrisi")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"{start_idx:02d}_correlation.png")
    plt.close()
    start_idx += 1

    # 21. Kapanış Fiyatı Line Plot
    plt.figure(figsize=(12, 5))
    plt.plot(df["date"], df["close"], label="Close", color="tab:blue")
    plt.title(f"{start_idx}. BTC Kapanış Fiyatı Zaman Serisi")
    plt.xlabel("Tarih")
    plt.ylabel("Fiyat ($)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"{start_idx:02d}_time_close.png")
    plt.close()
    start_idx += 1

    # 22. Volatilite Line Plot
    plt.figure(figsize=(12, 5))
    plt.plot(df["date"], df["volatility"], label="Volatility", color="tab:orange")
    plt.title(f"{start_idx}. BTC Volatilite Zaman Serisi")
    plt.xlabel("Tarih")
    plt.ylabel("Volatilite")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"{start_idx:02d}_time_volatility.png")
    plt.close()
    start_idx += 1

    # 23. Return vs Volatility Scatter
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x="return", y="volatility", hue="risk", alpha=0.6, palette="viridis")
    plt.title(f"{start_idx}. Getiri vs Volatilite (Risk Gruplarına Göre)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"{start_idx:02d}_scatter_ret_vol.png")
    plt.close()
    start_idx += 1

    # 24. Eksik Veri Çubuğu
    plt.figure(figsize=(10, 6))
    msno.bar(df, color=(0.2, 0.4, 0.6))
    plt.title(f"{start_idx}. Eksik Veri Analizi (Bar)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"{start_idx:02d}_missing_bar.png")
    plt.close()
    start_idx += 1

    # 25. Eksik Veri Matrisi
    plt.figure(figsize=(10, 6))
    msno.matrix(df, color=(0.2, 0.4, 0.6))
    plt.title(f"{start_idx}. Eksik Veri Analizi (Matrix)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"{start_idx:02d}_missing_matrix.png")
    plt.close()

def main():
    print("Veri yükleniyor...")
    df = load_data()
    print(f"Veri boyutu: {df.shape}")
    
    plot_all_graphics(df)
    print(f"Grafikler {FIGURES_DIR} klasörüne başarıyla kaydedildi.")

if __name__ == "__main__":
    main()
