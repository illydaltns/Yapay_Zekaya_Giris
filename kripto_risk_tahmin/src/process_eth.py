"""
process_eth.py

ETH için veri işleme ve model eğitimi pipeline'ı.
"""

from pathlib import Path
from pipeline_utils import process_coin_data, train_final_model

# ======================================================
# DİZİNLER
# ======================================================
BASE_DIR = Path(__file__).resolve().parent.parent

RAW_PATH = BASE_DIR / "data" / "raw" / "ETH_2019_2023_1d.csv"
TRAIN_PATH = BASE_DIR / "data" / "train" / "eth_train.csv"
TEST_PATH = BASE_DIR / "data" / "test" / "eth_test.csv"
MODEL_PATH = BASE_DIR / "models" / "eth_rf_risk_final_v1.pkl"

# ======================================================
# VERİ İŞLEME
# ======================================================
if RAW_PATH.exists():
    result = process_coin_data(
        coin_name="ETH",
        raw_data_path=RAW_PATH,
        train_path=TRAIN_PATH,
        test_path=TEST_PATH,
        split_date="2023-01-01"
    )
    
    # ======================================================
    # MODEL EĞİTİMİ
    # ======================================================
    if TRAIN_PATH.exists() and TEST_PATH.exists():
        train_final_model(
            coin_name="ETH",
            train_path=TRAIN_PATH,
            test_path=TEST_PATH,
            model_path=MODEL_PATH
        )
    else:
        print("❌ Train veya test dosyası bulunamadı!")
else:
    print(f"❌ Raw data dosyası bulunamadı: {RAW_PATH}")

