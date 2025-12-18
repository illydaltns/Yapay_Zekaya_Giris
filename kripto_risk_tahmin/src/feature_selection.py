"""
feature_selection.py

Bu dosyada, gerçekleştirilen EDA (Exploratory Data Analysis) sonuçlarına
dayanarak modelde kullanılacak nihai feature seti belirlenmiştir.

Bu aşamada:
- Herhangi bir model eğitimi yapılmaz
- Otomatik feature eleme uygulanmaz
- Sadece EDA'dan elde edilen kararlar koda yansıtılır

Amaç:
- Modeli gereksiz ve tekrarlı değişkenlerden arındırmak
- Daha sade, yorumlanabilir ve genellenebilir bir yapı elde etmek
"""

import pandas as pd
from pathlib import Path

# ======================================================
# Proje Dizinleri
# ======================================================
BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_PATH = BASE_DIR / "data" / "train" / "btc_train.csv"
TEST_PATH  = BASE_DIR / "data" / "test"  / "btc_test.csv"

# ======================================================
# EDA SONUÇLARINA GÖRE FEATURE SEÇİM KARARLARI
# ======================================================

"""
EDA sürecinde yapılan analizler sonucunda:

1) Korelasyon Analizi (Heatmap):
   - open, high, low ve close değişkenlerinin birbiriyle yüksek korelasyon
     gösterdiği tespit edilmiştir.
   - Bu nedenle fiyat bilgisini temsil etmesi açısından yalnızca `close`
     değişkeni tutulmuştur.

2) Türetilmiş Feature Analizi:
   - range değişkeninin volatility ile,
   - body değişkeninin return ile yüksek korelasyon gösterdiği gözlemlenmiştir.
   - Bilgi tekrarını önlemek amacıyla range ve body elenmiştir.

3) Risk ile Ayrıştırıcılık:
   - return ve volatility değişkenlerinin risk sınıfları arasında
     belirgin ayrışma sağladığı gözlemlenmiştir.
   - Bu değişkenler modelleme açısından anlamlı bulunmuştur.

4) Dağılım ve Gürültü Analizi:
   - Elenen değişkenlerin yüksek gürültü içerdiği ve
     modele ek katkı sağlamadığı değerlendirilmiştir.

5) Gerçek Hayat (Deploy) Uyumu:
   - Seçilen tüm feature'lar Binance API üzerinden
     gerçek zamanlı olarak üretilebilmektedir.
"""

# ======================================================
# NİHAİ FEATURE SETİ
# ======================================================

FINAL_FEATURES = [
    "close",        # Fiyatı temsil eden ana değişken
    "volume",       # Piyasa aktivitesini yansıtır (opsiyonel ama tutuldu)
    "return",       # Fiyat değişim oranı
    "volatility"    # Riskin temel göstergesi
]

TARGET = "risk"

# Bilinçli olarak elenen değişkenler (dokümantasyon amaçlı)
DROPPED_FEATURES = [
    "open",
    "high",
    "low",
    "range",
    "body"
]

# ======================================================
# VERİYİ OKU
# ======================================================
train_df = pd.read_csv(TRAIN_PATH)
test_df  = pd.read_csv(TEST_PATH)

# ======================================================
# FEATURE - TARGET AYRIMI
# ======================================================
X_train = train_df[FINAL_FEATURES]
y_train = train_df[TARGET]

X_test = test_df[FINAL_FEATURES]
y_test = test_df[TARGET] if TARGET in test_df.columns else None

# ======================================================
# KONTROL ÇIKTILARI
# ======================================================
print("=" * 60)
print("FEATURE SELECTION TAMAMLANDI")
print("=" * 60)

print("\nKullanılan Feature'lar:")
for f in FINAL_FEATURES:
    print(f"  - {f}")

print("\nElenen Feature'lar:")
for f in DROPPED_FEATURES:
    print(f"  - {f}")

print("\nVeri Boyutları:")
print("X_train:", X_train.shape)
print("X_test :", X_test.shape)

print("\nHazır → Model eğitim aşamasına geçilebilir.")
