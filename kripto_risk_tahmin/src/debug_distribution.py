
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def analyze_risk_distribution(coin_name, train_path, test_path):
    print(f"\n{'='*60}")
    print(f"ANALYZING RISK DISTRIBUTION FOR {coin_name}")
    print(f"{'='*60}")
    
    if not train_path.exists() or not test_path.exists():
        print(f"[ERROR] Files not found for {coin_name}")
        return

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # Calculate volatility manually to double check
    train_df["return"] = train_df["close"].pct_change()
    train_vol = train_df["return"].rolling(14).std()
    
    test_df["return"] = test_df["close"].pct_change()
    test_vol = test_df["return"].rolling(14).std()
    
    # Check original quantiles used for labeling
    # Note: The pipeline uses quantiles from the train set to label BOTH train and test.
    # We need to see what those quantiles are and how test data fits into them.
    
    q1 = train_vol.quantile(0.33)
    q2 = train_vol.quantile(0.66)
    
    print(f"\nTraining Set Quantiles (used for labeling):")
    print(f"Q1 (Low/Med Threshold): {q1:.6f}")
    print(f"Q2 (Med/High Threshold): {q2:.6f}")
    
    print(f"\nVolatility Stats:")
    print(f"Train Volatility Mean: {train_vol.mean():.6f}")
    print(f"Test Volatility Mean : {test_vol.mean():.6f}")
    
    print(f"\nActual Risk Labels in Files:")
    print(f"Train Risk Dist:\n{train_df['risk'].value_counts().sort_index()}")
    print(f"Test Risk Dist :\n{test_df['risk'].value_counts().sort_index()}")
    
    # Check what IF we recalculated quantiles on Test data (concept drift check)
    test_q1 = test_vol.quantile(0.33)
    test_q2 = test_vol.quantile(0.66)
    
    print(f"\n[CONCEPT DRIFT CHECK] Quantiles if calculated on Test Data:")
    print(f"Test Q1: {test_q1:.6f} (vs Train: {q1:.6f})")
    print(f"Test Q2: {test_q2:.6f} (vs Train: {q2:.6f})")
    
    if test_vol.mean() < train_vol.mean():
        print(f"\n[WARNING] Test data is significantly LESS volatile than Train data.")
        print("This explains why High Risk labels are missing/low in Test set.")
    
if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    data_dir = base_dir / "data"
    
    analyze_risk_distribution(
        "ETH", 
        data_dir / "train" / "eth_train.csv",
        data_dir / "test" / "eth_test.csv"
    )
    
    analyze_risk_distribution(
        "SOL", 
        data_dir / "train" / "sol_train.csv",
        data_dir / "test" / "sol_test.csv"
    )
