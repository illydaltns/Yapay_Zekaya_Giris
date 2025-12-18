# Proje İyileştirme Raporu ve Grafik Rehberi

Bu rapor, projenizdeki verileri anlamlandırmanız için üretilen **25 adet geliştirilmiş grafiği** ve yorumlarını içerir. Tüm grafikler profesyonel "darkgrid" temasıyla güncellendi.

## 1. Veri Dağılımı (Genel Bakış)
Veri setinizdeki hedef ve ana değişkenlerin genel durumunu gösterir.

### 1. Risk Sınıf Dağılımı
![Risk Dağılımı](figures/01_risk_distribution.png)
> **Yorum:** Veri setinizde dengesizlik var. Çoğu örnek "Yüksek Risk" (2) ve "Orta Risk" (1) grubunda. "Düşük Risk" (0) örnek sayısı çok az. Modelin düşük riski öğrenmesi zor olabilir.

---

## 2. OHLC (Açılış, Yüksek, Düşük, Kapanış, Hacim) Dağılımları
Fiyat verilerinin nasıl dağıldığını gösterir.

### 2. Open (Açılış) - 6. Volume (Hacim)
![Open](figures/02_open_hist.png)
![High](figures/03_high_hist.png)
![Low](figures/04_low_hist.png)
![Close](figures/05_close_hist.png)
![Volume](figures/06_volume_hist.png)

> **Yorum:**
> - Fiyatlar (Open, High, Low, Close) genellikle benzer dağılım gösterir (sağa çarpık olabilir). Bu normaldir.
> - **Volume (Hacim):** Genelde log-normal dağılır. Çok yüksek hacimli günler "outlier" (aykırı değer) olabilir ve önemli piyasa hareketlerini işaret eder.

---

## 3. Türetilmiş Özellik Dağılımları
Hesaplanan feature'ların (Return, Volatility vb.) dağılımı.

### 7. Return (Getiri) - 10. Body (Gövde)
![Return](figures/07_return_hist.png)
![Volatility](figures/08_volatility_hist.png)
![Range](figures/09_range_hist.png)
![Body](figures/10_body_hist.png)

> **Yorum:**
> - **Return:** Sıfır etrafında çan eğrisi (Normal dağılım) oluşturması beklenir. Kuyrukların uzunluğu risk iştahını gösterir.
> - **Volatility:** Genelde sağa çarpıktır. Yüksek volatilite yüksek risk demektir.

---

## 4. Risk Gruplarına Göre OHLC Analizi (Boxplot)
Fiyatların risk gruplarına göre nasıl değiştiğini gösterir.

### 11. Open - 15. Volume
![Open Box](figures/11_open_boxplot.png)
![High Box](figures/12_high_boxplot.png)
![Low Box](figures/13_low_boxplot.png)
![Close Box](figures/14_close_boxplot.png)
![Volume Box](figures/15_volume_boxplot.png)

> **Yorum:**
> - Kutuların (box) boyutu ve "çizgilerin" (whisker) uzunluğu o risk grubundaki değişkenliği gösterir.
> - Eğer bir risk grubunda kutu çok daha yukarıdaysa, o risk seviyesinde fiyat/hacim daha yüksek demektir.

---

## 5. Risk Gruplarına Göre Türetilmiş Özellikler (Boxplot)
**En önemli analizler buradadır.** Risk ile doğrudan ilişkili özellikleri gösterir.

### 16. Return - 19. Body
![Return Box](figures/16_return_boxplot.png)
![Volatility Box](figures/17_volatility_boxplot.png)
![Range Box](figures/18_range_boxplot.png)
![Body Box](figures/19_body_boxplot.png)

> **Yorum:**
> - **Volatility vs Risk:** Yüksek risk grubunda (2), Volatility kutusunun çok daha yukarıda olmasını beklersiniz. Zaten etiketleme buna göre yapıldıysa bu bir doğrulama grafiğidir.
> - **Range (Yüksek-Düşük farkı):** Yüksek riskli günlerde fiyat aralığının (Range) daha geniş olması beklenir.

---

## 6. İlişkiler ve Zaman Serisi

### 20. Korelasyon Matrisi
![Korelasyon](figures/20_correlation.png)
> **Yorum:** Değişkenlerin birbirleri ile ilişkisi.
> - Kırmızı (1.0): Pozitif güçlü ilişki.
> - Mavi (-1.0): Negatif güçlü ilişki.
> - **Önemli:** `Risk` satırına bakın. En yüksek korelasyon hangi özellik ile? (Muhtemelen Volatility).

### 21. Fiyat ve 22. Volatilite Zaman Serisi
![Fiyat](figures/21_time_close.png)
![Volatilite](figures/22_time_volatility.png)

> **Yorum:**
> - Fiyatın zamanla değişimi.
> - Volatilitenin arttığı dönemler (turuncu grafik tepeler), riskli dönemlerdir.

### 23. Getiri vs Volatilite (Scatter)
![Scatter](figures/23_scatter_ret_vol.png)
> **Yorum:**
> - Yatay eksen getiri, dikey eksen volatilite.
> - Renkler risk grubunu gösterir. Sarı noktaların (Yüksek Risk) genellikle grafiğin üst kısmında toplanması gerekir.

---

## 7. Veri Kalitesi

### 24. Eksik Veri (Bar) ve 25. Matris
![Missing Bar](figures/24_missing_bar.png)
![Missing Matrix](figures/25_missing_matrix.png)

> **Yorum:**
> - Eğer çubuklar tam doluysa (en üst seviyede), verinizde eksik yok demektir.
> - Beyaz çizgiler eksik verileri temsil eder.

---

## Model Sonuçları
Mevcut modelinizin performansı:
![Confusion Matrix](figures/confusion_matrix.png)
> Model ağırlıklı olarak "Yüksek Risk" tahmin ediyor. Veri dengeleme çalışması yapılması önerilir.
