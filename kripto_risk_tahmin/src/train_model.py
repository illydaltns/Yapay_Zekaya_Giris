import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# =========================
# Ayarlar
# =========================
sns.set_theme(style="darkgrid", context="talk", palette="viridis")
BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN_PATH = BASE_DIR / "data" / "train" / "btc_train.csv"
MODEL_DIR = BASE_DIR / "models"
FIGURES_DIR = BASE_DIR / "reports" / "figures"

# Klasörleri oluştur
MODEL_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

def train_and_evaluate():
    print("Veri yükleniyor...")
    df = pd.read_csv(TRAIN_PATH)
    
    FEATURES = ["return", "range", "body", "volume"]
    TARGET = "risk"
    
    X = df[FEATURES]
    y = df[TARGET]
    
    # Shuffle=False zaman serisi olduğu için (genelde)
    # Ancak orijinal kodda Feature Engineering kısmında zaten train/test ayrılmıştı.
    # Burada train içinden validasyon ayırıyoruz.
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    
    print("Model eğitiliyor...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        random_state=42,
        class_weight="balanced"
    )
    model.fit(X_train, y_train)
    
    # Validation Tahminleri
    y_pred = model.predict(X_val)
    
    # Metrikler
    acc = accuracy_score(y_val, y_pred)
    print(f"\nValidation Accuracy: {acc:.4f}")
    
    print("\nSınıflandırma Raporu:")
    print(classification_report(y_val, y_pred, target_names=["Düşük", "Orta", "Yüksek"]))
    
    # --- Confusion Matrix ---
    cm = confusion_matrix(y_val, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=["Düşük", "Orta", "Yüksek"],
                yticklabels=["Düşük", "Orta", "Yüksek"])
    plt.title("Karmaşıklık Matrisi (Confusion Matrix)")
    plt.xlabel("Tahmin Edilen")
    plt.ylabel("Gerçek")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "confusion_matrix.png")
    print(f"Confusion matrix kaydedildi: {FIGURES_DIR / 'confusion_matrix.png'}")
    # plt.show() # Otomasyon için kapalı tutuyoruz
    plt.close()

    # --- Feature Importance ---
    importances = model.feature_importances_
    indices = importances.argsort()[::-1]
    feature_names = [FEATURES[i] for i in indices]
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances[indices], y=feature_names, palette="viridis")
    plt.title("Özellik Önem Düzeyleri (Feature Importance)")
    plt.xlabel("Önem Skoru")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "feature_importance.png")
    print(f"Feature importance kaydedildi: {FIGURES_DIR / 'feature_importance.png'}")
    # plt.show()
    plt.close()
    
    # Modeli Kaydet
    model_path = MODEL_DIR / "risk_model_pro.pkl"
    joblib.dump(model, model_path)
    print(f"\nModel kaydedildi: {model_path}")

if __name__ == "__main__":
    train_and_evaluate()
