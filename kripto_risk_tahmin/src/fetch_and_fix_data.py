import pandas as pd
from pathlib import Path
from binance.client import Client
from datetime import datetime
import time

def fetch_historical_data(symbol, start_str, end_str, output_path):
    print(f"\n[INFO] Fetching {symbol} data from {start_str} to {end_str}...")
    
    client = Client()
    interval = Client.KLINE_INTERVAL_1DAY
    
    try:
        klines = client.get_historical_klines(symbol, interval, start_str, end_str)
        
        if not klines:
            print(f"[ERROR] No data found for {symbol}")
            return
            
        print(f"[INFO] Downloaded {len(klines)} days of data.")
        
        df = pd.DataFrame(klines, columns=[
            "date", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume",
            "number_of_trades", "taker_buy_base",
            "taker_buy_quote", "ignore"
        ])
        
        df["date"] = pd.to_datetime(df["date"], unit="ms")
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)
            
        final_df = df[["date", "open", "high", "low", "close", "volume"]]
        final_df.to_csv(output_path, index=False)
        
        print(f"[OK] Saved to {output_path}")
        
    except Exception as e:
        print(f"[ERROR] Failed to fetch {symbol}: {e}")


def run():
    base_dir = Path(__file__).resolve().parent.parent
    raw_dir = base_dir / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    fetch_historical_data(
        "ETHUSDT", 
        "1 Jan, 2019", 
        "31 Dec, 2023", 
        raw_dir / "ETH_2019_2023_1d.csv"
    )
    
    fetch_historical_data(
        "SOLUSDT", 
        "1 Jan, 2019", 
        "31 Dec, 2023", 
        raw_dir / "SOL_2019_2023_1d.csv"
    )


run()
