"""
pipeline_utils.py

Coin-agnostic pipeline fonksiyonları.
Tüm coinler için ortak kullanılabilecek fonksiyonlar.
"""

import pandas as pd
from pathlib import Path
from typing import Tuple, Dict
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering fonksiyonu.
    Tüm coinler için aynı feature'ları oluşturur.
    """
    df = df.sort_values("date").copy()

    # Temel feature'lar
    df["return"] = df["close"].pct_change()
    df["volatility"] = df["return"].rolling(window=14).std()
    df["range"] = df["high"] - df["low"]
    df["body"] = df["close"] - df["open"]

    # Gecikmeli getiriler (momentum)
    df["return_lag_1"] = df["return"].shift(1)
    df["return_lag_3"] = df["return"].shift(3)

    # Hareketli ortalamadan sapma
    df["ma_7"] = df["close"].rolling(7).mean()
    df["ma_7_diff"] = (df["close"] - df["ma_7"]) / df["ma_7"]

    # Oransal günlük aralık
    df["range_pct"] = df["range"] / df["close"]

    return df.dropna()


def create_risk_labels(df: pd.DataFrame, quantiles: Tuple[float, float] = None) -> pd.DataFrame:
    """
    Volatiliteye göre risk etiketleri oluşturur.
    
    Args:
        df: Feature engineering yapılmış dataframe
        quantiles: (q1, q2) tuple. None ise train'den hesaplanır.
    
    Returns:
        Risk etiketleri eklenmiş dataframe ve quantile değerleri
    """
    if quantiles is None:
        q1, q2 = df["volatility"].quantile([0.33, 0.66])
    else:
        q1, q2 = quantiles

    def risk_label(v):
        if pd.isna(v):
            return 1  # Default medium risk
        if v <= q1:
            return 0   # Low risk
        elif v <= q2:
            return 1   # Medium risk
        else:
            return 2   # High risk

    df["risk"] = df["volatility"].apply(risk_label)
    return df, (q1, q2)


def create_dynamic_risk_labels(df: pd.DataFrame, window: int = 90, threshold: float = 1.0) -> pd.DataFrame:
    """
    Dinamik Z-Score bazlı risk etiketleme.
    Konsept kaymasını (Concept Drift) yönetmek için kullanılır.
    
    Args:
        df: Feature engineering yapılmış dataframe
        window: Z-Score için rolling window (gün)
        threshold: Yüksek risk için Z-Score eşiği
        
    Returns:
        Risk etiketleri güncellenmiş dataframe
    """
    df = df.copy()
    
    # Rolling Mean ve Std
    roll_mean = df["volatility"].rolling(window=window, min_periods=window//2).mean()
    roll_std = df["volatility"].rolling(window=window, min_periods=window//2).std()
    
    # Z-Score
    df["vol_z_score"] = (df["volatility"] - roll_mean) / roll_std
    
    def z_risk_label(z):
        if pd.isna(z):
            return 1 # Default Medium
        if z <= -0.5:
            return 0 # Low Risk (Ortalamanın altında)
        elif z <= threshold:
            return 1 # Medium Risk
        else:
            return 2 # High Risk (Ortalamadan 1 std sapma yukarıda)
            
    df["risk"] = df["vol_z_score"].apply(z_risk_label)
    return df


def split_train_test(df: pd.DataFrame, split_date: str = "2023-01-01") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Zaman bazlı train/test split.
    
    Args:
        df: Tarih sütunu içeren dataframe
        split_date: Split tarihi (string format: "YYYY-MM-DD")
    
    Returns:
        train_df, test_df
    """
    df["date"] = pd.to_datetime(df["date"])
    train_df = df[df["date"] < split_date].copy()
    test_df = df[df["date"] >= split_date].copy()
    return train_df, test_df


def process_coin_data(
    coin_name: str,
    raw_data_path: Path,
    train_path: Path,
    test_path: Path,
    split_date: str = "2023-01-01"
) -> Dict:
    """
    Bir coin için tüm veri işleme pipeline'ını çalıştırır.
    
    Args:
        coin_name: Coin adı (örn: "BTC", "ETH", "SOL")
        raw_data_path: Ham veri dosya yolu
        train_path: Train verisi kayıt yolu
        test_path: Test verisi kayıt yolu
        test_path: Test verisi kayıt yolu
        split_date: Split tarihi
        use_dynamic_labeling: True ise dinamik Z-score kullanır
    
    Returns:
        İşlem sonuçları dictionary
    """
    print(f"\n{'='*70}")
    print(f"PROCESSING {coin_name.upper()} DATA")
    print(f"{'='*70}")
    
    # Use dynamic labeling for ETH and SOL (known concept drift), static for BTC (baseline)
    # Or make it configurable. For now, let's hardcode it for optimization task or pass as arg.
    # To keep func signature clean, we will determine strategy inside or use default arg.
    use_dynamic_labeling = True if coin_name in ["ETH", "SOL"] else False
    if use_dynamic_labeling:
        print(f"[INFO] Using DYNAMIC Risk Labeling (Rolling Z-Score) for {coin_name}")
    else:
        print(f"[INFO] Using STATIC Quantile Labeling for {coin_name}")
    
    # 1. Ham veriyi oku
    print(f"\n1. Reading raw data from: {raw_data_path}")
    try:
        df = pd.read_csv(raw_data_path, index_col=0)
    except:
        df = pd.read_csv(raw_data_path)
    
    # Tarih sütununu standartlaştır
    # Eğer date/datetime index'e gittiyse geri al
    if "date" not in df.columns and "datetime" not in df.columns:
        df = df.reset_index()

    if "datetime" in df.columns:
        df = df.rename(columns={"datetime": "date"})
    
    # Hala date yoksa hata vermeden önce kontrol et
    if "date" not in df.columns:
        # Index'in kendisi tarih olabilir ama ismi olmayabilir
        df = df.reset_index()
        # Reset sonrası ilk (veya 'index') kolon tarihi tutuyor olabilir, rename et
        if "index" in df.columns:
            df = df.rename(columns={"index": "date"})
        
    df["date"] = pd.to_datetime(df["date"])
    
    print(f"   Raw data shape: {df.shape}")
    
    # 2. Train/Test split
    print(f"\n2. Splitting data (split date: {split_date})")
    train_df, test_df = split_train_test(df, split_date)
    print(f"   Train shape: {train_df.shape}")
    print(f"   Test shape: {test_df.shape}")
    
    # 3. Feature engineering - Train
    print(f"\n3. Feature engineering (train)")
    train_df = add_features(train_df)
    
    # 4. Risk labels - Train
    print(f"\n4. Creating risk labels (train)")
    
    quantiles = None
    if use_dynamic_labeling:
        train_df = create_dynamic_risk_labels(train_df)
        print("   Using dynamic Z-Score labeling (no fixed quantiles)")
    else:
        train_df, quantiles = create_risk_labels(train_df)
        q1, q2 = quantiles
        print(f"   Risk quantiles: q1={q1:.6f}, q2={q2:.6f}")
        
    print(f"   Risk distribution:")
    print(train_df["risk"].value_counts().sort_index())
    
    # 5. Feature engineering - Test
    print(f"\n5. Feature engineering (test)")
    test_df = add_features(test_df)
    
    # 6. Risk labels - Test
    print(f"\n6. Creating risk labels (test)")
    
    if use_dynamic_labeling:
        # Test içinde de kendi rolling window'unu kullanır (Data Leakage olmaz çünkü rolling past veriyi kullanır)
        # Ancak test verisinin başındaki rolling değerleri için train'in sonunu eklemek gerekir.
        # Basitlik için test setinde kendi içinde hesaplayalım (Validation için kabul edilebilir)
        # Production için: Train'in son window'unu test'in başına ekleyip hesaplamak lazım.
        
        # Daha sağlam yöntem: Full datayı label'la sonra böl.
        # Ancak mevcut yapı split -> label şeklindeydi.
        # Dinamik labeling için ÖNCE label sonra split daha mantıklı.
        # Bu fonksiyonu biraz refactor edelim.
        pass # Aşağıda logic değişecek.
    else:
        test_df, _ = create_risk_labels(test_df, quantiles=quantiles)
        
    # REFACTORING: Dynamic Labeling needs continuous data to calculate rolling stats correctly at the boundary.
    # So we should Apply Labeling -> Then Split if dynamic.
    
    if use_dynamic_labeling:
        # Full df üzerinde uygula (önce featureları ekle)
        # Not: add_features zaten çağrılmıştı partial dfler için.
        # En iyisi: raw data -> add features -> dynamic label -> split
        print("   Refactoring flow for Dynamic Labeling...")
        full_df = add_features(df) # Recalculate robust features on full data
        full_df = create_dynamic_risk_labels(full_df)
        
        # Split again
        train_df, test_df = split_train_test(full_df, split_date)
        
    print(f"   Test Risk Method: {'Dynamic' if use_dynamic_labeling else 'Static (Train Quantiles)'}")
    print(f"   Risk distribution (Test):")
    print(test_df["risk"].value_counts().sort_index())
    
    # 7. Kaydet
    print(f"\n7. Saving processed data")
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    print(f"   Train saved: {train_path}")
    print(f"   Test saved: {test_path}")
    
    return {
        "coin_name": coin_name,
        "train_shape": train_df.shape,
        "test_shape": test_df.shape,
        "quantiles": quantiles,
        "train_risk_dist": train_df["risk"].value_counts().to_dict(),
        "test_risk_dist": test_df["risk"].value_counts().to_dict()
    }


def train_final_model(
    coin_name: str,
    train_path: Path,
    test_path: Path,
    model_path: Path,
    features: list = None
) -> Dict:
    """
    Final Random Forest modelini eğitir ve değerlendirir.
    
    Args:
        coin_name: Coin adı
        train_path: Train verisi yolu
        test_path: Test verisi yolu
        model_path: Model kayıt yolu
        features: Kullanılacak feature listesi (None ise default)
    
    Returns:
        Performans metrikleri dictionary
    """
    if features is None:
        features = [
            "close",
            "volume",
            "return",
            "body",
            "range",
            "return_lag_1",
            "return_lag_3",
            "ma_7_diff",
            "range_pct"
        ]
    
    print(f"\n{'='*70}")
    print(f"TRAINING FINAL MODEL FOR {coin_name.upper()}")
    print(f"{'='*70}")
    
    # Veriyi oku
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    X_train = train_df[features]
    y_train = train_df["risk"]
    X_test = test_df[features]
    y_test = test_df["risk"]
    
    print(f"\nFeatures: {features}")
    print(f"Train shape: {X_train.shape}")
    print(f"Test shape: {X_test.shape}")
    
    # Model oluştur ve eğit
    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=6,
        min_samples_leaf=3,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )
    
    print(f"\nTraining model...")
    model.fit(X_train, y_train)
    
    # Train performansı
    train_pred = model.predict(X_train)
    train_acc = accuracy_score(y_train, train_pred)
    
    # Test performansı
    test_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, test_pred)
    
    print(f"\n{'='*70}")
    print("PERFORMANCE METRICS")
    print(f"{'='*70}")
    print(f"\nTrain Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    
    print(f"\nClassification Report (Test):")
    print(classification_report(y_test, test_pred))
    
    print(f"\nConfusion Matrix (Test):")
    print(confusion_matrix(y_test, test_pred))
    
    # Modeli kaydet
    model_path.parent.mkdir(exist_ok=True, parents=True)
    joblib.dump(model, model_path)
    print(f"\n[OK] Model saved: {model_path}")
    
    return {
        "coin_name": coin_name,
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "features": features,
        "model_path": str(model_path)
    }

