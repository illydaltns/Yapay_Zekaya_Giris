import pandas as pd
import numpy as np
from pathlib import Path
import json
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, f1_score

# ======================================================
# ARAÇLAR VE DİZİNLER
# ======================================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "train"
OUTPUT_FILE = BASE_DIR / "src" / "coin_params.json"

# Ortak Feature Set
FEATURES = [
    "close", "volume", "return", "body", "range",
    "return_lag_1", "return_lag_3", "ma_7_diff", "range_pct"
]

def optimize_for_coin(coin_name, train_path):
    print(f"\n{'='*70}")
    print(f"OPTIMIZING HYPERPARAMETERS FOR {coin_name}")
    print(f"{'='*70}")
    
    if not train_path.exists():
        print(f"[ERROR] Train file not found: {train_path}")
        return None
        
    df = pd.read_csv(train_path)
    X = df[FEATURES]
    y = df["risk"]
    
    print(f"Data Shape: {X.shape}")
    print("Risk Distribution:")
    print(y.value_counts().sort_index())
    
    param_grid = {
        'n_estimators': [200, 400],
        'max_depth': [4, 6, 8, 10],
        'min_samples_leaf': [3, 5, 10],
        'class_weight': ['balanced', 'balanced_subsample']
    }
    
    tscv = TimeSeriesSplit(n_splits=3)
    f1_macro = make_scorer(f1_score, average='macro')
    
    grid_search = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42, n_jobs=-1),
        param_grid=param_grid,
        cv=tscv,
        scoring=f1_macro,
        verbose=1,
        n_jobs=-1
    )
    
    print("\nRunning Grid Search...")
    grid_search.fit(X, y)
    
    print("\n✅ Optimization Completed!")
    print(f"Best Score (Macro F1): {grid_search.best_score_:.4f}")
    print(f"Best Params: {grid_search.best_params_}")
    
    return grid_search.best_params_

# ======================================================
# DOĞRUDAN ÇALIŞAN AKIŞ
# ======================================================

coins_to_optimize = {
    "ETH": DATA_DIR / "eth_train.csv",
    "SOL": DATA_DIR / "sol_train.csv",
    "BTC": DATA_DIR / "btc_train.csv"
}

best_params = {}

for coin, path in coins_to_optimize.items():
    params = optimize_for_coin(coin, path)
    if params:
        best_params[coin] = params

with open(OUTPUT_FILE, "w") as f:
    json.dump(best_params, f, indent=4)

print(f"\n[OK] All robust parameters saved to {OUTPUT_FILE}")
