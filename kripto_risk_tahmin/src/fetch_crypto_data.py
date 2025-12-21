import ccxt
import pandas as pd
from datetime import datetime
import os

def fetch_data(symbol, start_date, end_date, timeframe='1d'):
    """
    Belirtilen coin için tarih aralığında veri çeker ve CSV olarak kaydeder.
    
    Args:
        symbol (str): Coin çifti (örn: 'ETH/USDT', 'BTC/USDT', 'SOL/USDT')
        start_date (str): Başlangıç tarihi (YYYY-MM-DD formatında)
        end_date (str): Bitiş tarihi (YYYY-MM-DD formatında)
        timeframe (str): Zaman aralığı (varsayılan '1d')
    """
    exchange = ccxt.binance()
    
    # Tarih aralığı: ccxt timestamp bekler (ms)
    # Z saati (UTC) ekleyerek parse ediyoruz
    since = exchange.parse8601(f'{start_date}T00:00:00Z')
    end_time_ms = exchange.parse8601(f'{end_date}T23:59:59Z')
    
    all_ohlcv = []
    
    print(f"Veri çekiliyor: {symbol} ({timeframe})")
    print(f"Başlangıç: {start_date}, Bitiş: {end_date}")
    
    current_since = since
    
    while current_since < end_time_ms:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, current_since, limit=1000)
            if not ohlcv:
                break
            
            all_ohlcv.extend(ohlcv)
            current_since = ohlcv[-1][0] + 1  # Sonraki mumun zamanına geç
            
            # Bitiş tarihini geçtik mi?
            if current_since > end_time_ms:
                break
                
            # Hızlı ilerlemeyi göstermek için nokta basabiliriz
            print(".", end="", flush=True)
            
        except Exception as e:
            print(f"\nHata: {e}")
            break
            
    print("\nVeri çekme tamamlandı. İşleniyor...")
    
    if not all_ohlcv:
        print("Veri bulunamadı.")
        return

    # DataFrame oluşturma
    # Sütunlar: timestamp, open, high, low, close, volume (ccxt standardı)
    df = pd.DataFrame(all_ohlcv, columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
    
    # Tarih formatı ve filtreleme
    df['datetime'] = pd.to_datetime(df['datetime'], unit='ms')
    
    # İstenen tarih aralığını kesinleştirme
    mask = (df['datetime'] >= start_date) & (df['datetime'] <= end_date)
    df = df.loc[mask].copy()
    
    if df.empty:
        print("Belirtilen tarih aralığında uygun veri yok.")
        return

    # Tarih formatı string olarak YYYY-MM-DD
    df['datetime'] = df['datetime'].dt.strftime('%Y-%m-%d')
    
    # Index sütunu 0'dan başlayacak şekilde reset
    df.reset_index(drop=True, inplace=True)
    df.index.name = 'index' 
    
    # İstenen sütun sıralaması: index, datetime, open, high, low, close, volume
    cols = ['datetime', 'open', 'high', 'low', 'close', 'volume']
    df = df[cols]
    
    # Dinamik dosya ismi oluşturma
    # Symbol '/' içeriyor, dosya isminde '_' yapalım
    safe_symbol = symbol.replace('/', '_').split('_')[0] # Sadece ilk kısmı alalım (ETH, SOL vs.)
    start_year = start_date.split('-')[0]
    end_year = end_date.split('-')[0]
    
    output_filename = f"{safe_symbol}_{start_year}_{end_year}_{timeframe}.csv"
    
    # Kaydetme yolu
    # Bu scriptin src altında olduğunu varsayıyoruz, bir üst klasördeki data/raw'a gideceğiz
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # src'den bir yukarı çık -> kripto_risk_tahmin -> data -> raw
    project_root = os.path.dirname(current_dir)
    output_dir = os.path.join(project_root, 'data', 'raw')
    
    os.makedirs(output_dir, exist_ok=True)
    output_file_path = os.path.join(output_dir, output_filename)
    
    # CSV kaydetme
    # index sütununu verisetine dahil ediyoruz
    df_final = df.reset_index()
    df_final.rename(columns={'index': 'index'}, inplace=True)
    
    df_final.to_csv(output_file_path, index=False)
    
    print(f"Dosya kaydedildi: {output_file_path}")
    print("İlk 5 satır:")
    print(df_final.head())
