# Yapay_Zekaya_Giris

## 🎯 Proje Özellikleri

- **Çoklu Coin Desteği**: BTC, ETH, BNB ve diğer popüler coinler
- **Çoklu Model Karşılaştırması**: Random Forest, Logistic Regression, SVM, KNN, Decision Tree
- **Otomatik Model Seçimi**: En iyi performans gösteren modeller otomatik seçilir
- **Binance API Entegrasyonu**: Gerçek zamanlı veri çekme
- **Web Arayüzü**: Streamlit ile kullanıcı dostu arayüz
- **Risk Sınıflandırması**: Düşük, Orta, Yüksek risk seviyeleri

## 📁 Proje Yapısı

```
kripto_risk_tahmin/
├── data/
│   ├── raw/          # Ham veriler (Binance'den çekilen)
│   ├── train/        # Eğitim verileri
│   └── test/         # Test verileri
├── models/           # Eğitilmiş modeller (coin bazında)
│   ├── BTC/
│   ├── ETH/
│   └── ...
├── src/
│   ├── data_fetcher.py          # Binance API entegrasyonu
│   ├── split_data.py            # Veri bölme
│   ├── feature_engineering.py  # Özellik mühendisliği
│   ├── model_trainer.py         # Model eğitimi ve karşılaştırma
│   ├── predictor.py             # Tahmin modülü
│   └── pipeline.py               # Tüm pipeline'ı çalıştırma
└── app.py            # Streamlit web arayüzü
```

## 🚀 Kurulum

### 1. Gereksinimleri Yükleyin

```bash
pip install -r requirements.txt
```

### 2. Binance API (Opsiyonel)

Public veri için API key gerekmez. Ancak rate limit'leri artırmak için API key kullanabilirsiniz.

## 📖 Kullanım

### Adım 1: Veri Çekme ve Model Eğitimi

Bir coin için tüm pipeline'ı çalıştırmak:

```python
from kripto_risk_tahmin.src.pipeline import process_coin

# BTC için veri çek, işle ve modelleri eğit
process_coin('BTCUSDT', start_date='2019-01-01')
```

Veya komut satırından:

```bash
cd kripto_risk_tahmin/src
python pipeline.py
```

### Adım 2: Web Arayüzünü Başlatma

```bash
streamlit run kripto_risk_tahmin/app.py
```

Tarayıcınızda `http://localhost:8501` adresine gidin.

### Adım 3: Arayüzde Kullanım

1. Sol menüden coin seçin
2. "Güncel Veri Çek" butonuna tıklayın (ilk kullanımda)
3. Pipeline'ı çalıştırarak modelleri eğitin
4. Risk analizini görüntüleyin

## 🔧 Modül Kullanımı

### Veri Çekme

```python
from kripto_risk_tahmin.src.data_fetcher import BinanceDataFetcher

fetcher = BinanceDataFetcher()
fetcher.save_coin_data('BTCUSDT', start_date='2019-01-01')
```

### Model Eğitimi

```python
from kripto_risk_tahmin.src.model_trainer import ModelTrainer

trainer = ModelTrainer(coin_name='BTC')
trainer.train_all_models('data/train/btc_train.csv')
best_models = trainer.get_best_models(top_n=3)
```

### Tahmin Yapma

```python
from kripto_risk_tahmin.src.predictor import RiskPredictor
import pandas as pd

# Veriyi yükle
df = pd.read_csv('data/test/btc_test.csv')
df['date'] = pd.to_datetime(df['date'])

# Tahmin yap
predictor = RiskPredictor('BTC')
result = predictor.predict(df.tail(30), top_n=3)
print(result)
```

## 📊 Risk Sınıflandırması

Risk seviyeleri volatiliteye göre belirlenir:

- **🟢 Düşük Risk (0)**: Volatilite alt %33
- **🟡 Orta Risk (1)**: Volatilite %33-%66
- **🔴 Yüksek Risk (2)**: Volatilite üst %33

## 🤖 Kullanılan Modeller

1. **Random Forest**: Ensemble yöntemi, güçlü performans
2. **Logistic Regression**: Hızlı ve yorumlanabilir
3. **SVM (Support Vector Machine)**: Karmaşık sınırlar için
4. **KNN (K-Nearest Neighbors)**: Basit ve etkili
5. **Decision Tree**: Yorumlanabilir karar ağaçları

## 📈 Özellikler

Model eğitimi için kullanılan özellikler:

- **return**: Günlük getiri (close fiyatının yüzde değişimi)
- **range**: High - Low (günlük fiyat aralığı)
- **body**: Close - Open (mum gövdesi)
- **volume**: İşlem hacmi

## 🎯 Gelecek Geliştirmeler

- [ ] Daha fazla teknik indikatör ekleme (RSI, MACD, vb.)
- [ ] Deep Learning modelleri (LSTM, GRU)
- [ ] Portföy risk analizi
- [ ] E-posta/Telegram bildirimleri
- [ ] Backtesting sistemi
- [ ] Model otomatik yeniden eğitimi

## 📝 Notlar

- İlk kullanımda modelleri eğitmek zaman alabilir
- Binance API rate limit'lerine dikkat edin
- Model performansları coin bazında değişebilir
- Veri kalitesi model performansını etkiler

## 🤝 Katkıda Bulunma

Projeye katkıda bulunmak için:

1. Fork yapın
2. Feature branch oluşturun (`git checkout -b feature/amazing-feature`)
3. Commit yapın (`git commit -m 'Add amazing feature'`)
4. Push yapın (`git push origin feature/amazing-feature`)
5. Pull Request açın

## 📄 Lisans

Bu proje eğitim amaçlıdır.

## 👤 Yazar

Yapay Zekaya Giriş Projesi

---

**⚠️ Uyarı**: Bu proje sadece eğitim amaçlıdır. Yatırım tavsiyesi değildir. Kripto para yatırımları risklidir.
