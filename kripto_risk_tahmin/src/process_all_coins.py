import ccxt
import pandas as pd
from datetime import datetime
import os

def fetch_data(symbol, start_date, end_date, timeframe='1d'):
    """
    Belirtilen coin için tarih aralığında veri çeker ve CSV olarak kaydeder.
    """
    exchange = ccxt.binance()
    
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
            current_since = ohlcv[-1][0] + 1
            
            if current_since > end_time_ms:
                break
                
            print(".", end="", flush=True)
            
        except Exception as e:
            print(f"\nHata: {e}")
            break
            
    print("\nVeri çekme tamamlandı. İşleniyor...")
    
    if not all_ohlcv:
        print("Veri bulunamadı.")
        return

    df = pd.DataFrame(
        all_ohlcv,
        columns=['datetime', 'open', 'high', 'low', 'close', 'volume']
    )
    
    df['datetime'] = pd.to_datetime(df['datetime'], unit='ms')
    
    mask = (df['datetime'] >= start_date) & (df['datetime'] <= end_date)
    df = df.loc[mask].copy()
    
    if df.empty:
        print("Belirtilen tarih aralığında uygun veri yok.")
        return

    df['datetime'] = df['datetime'].dt.strftime('%Y-%m-%d')
    df.reset_index(drop=True, inplace=True)
    df.index.name = 'index'
    
    df = df[['datetime', 'open', 'high', 'low', 'close', 'volume']]
    
    safe_symbol = symbol.replace('/', '_').split('_')[0]
    start_year = start_date.split('-')[0]
    end_year = end_date.split('-')[0]
    
    output_filename = f"{safe_symbol}_{start_year}_{end_year}_{timeframe}.csv"
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    output_dir = os.path.join(project_root, 'data', 'raw')
    
    os.makedirs(output_dir, exist_ok=True)
    output_file_path = os.path.join(output_dir, output_filename)
    
    df_final = df.reset_index()
    df_final.to_csv(output_file_path, index=False)
    
    print(f"Dosya kaydedildi: {output_file_path}")
    print(df_final.head())


def run():
    fetch_data("ETH/USDT", "2019-01-01", "2023-12-31", "1d")
    fetch_data("SOL/USDT", "2019-01-01", "2023-12-31", "1d")


run()
