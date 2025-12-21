"""
app.py

Streamlit web uygulaması - Kripto Risk Tahmin Sistemi
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from binance.client import Client

# Sayfa konfigürasyonu
st.set_page_config(
    page_title="Kripto Risk Tahmin Sistemi",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================================================
# DİZİNLER VE YÜKLEMELER
# ======================================================
BASE_DIR = Path(__file__).resolve().parent

# Coin konfigürasyonları
COINS = {
    "BTC": {
        "name": "Bitcoin",
        "symbol": "BTCUSDT",
        "train": BASE_DIR / "data" / "train" / "btc_train.csv",
        "test": BASE_DIR / "data" / "test" / "btc_test.csv",
        "model": BASE_DIR / "models" / "rf_risk_final_v1.pkl"
    },
    "ETH": {
        "name": "Ethereum",
        "symbol": "ETHUSDT",
        "train": BASE_DIR / "data" / "train" / "eth_train.csv",
        "test": BASE_DIR / "data" / "test" / "eth_test.csv",
        "model": BASE_DIR / "models" / "eth_rf_risk_final_v1.pkl"
    },
    "SOL": {
        "name": "Solana",
        "symbol": "SOLUSDT",
        "train": BASE_DIR / "data" / "train" / "sol_train.csv",
        "test": BASE_DIR / "data" / "test" / "sol_test.csv",
        "model": BASE_DIR / "models" / "sol_rf_risk_final_v1.pkl"
    }
}

# Risk etiketleri
RISK_LABELS = {
    0: {"label": "Düşük Risk", "color": "🟢", "color_hex": "#00ff00"},
    1: {"label": "Orta Risk", "color": "🟡", "color_hex": "#ffff00"},
    2: {"label": "Yüksek Risk", "color": "🔴", "color_hex": "#ff0000"}
}

# ======================================================
# YARDIMCI FONKSİYONLAR
# ======================================================

@st.cache_data
def load_model(coin: str):
    """Modeli yükle (cache'lenmiş)"""
    model_path = COINS[coin]["model"]
    if model_path.exists():
        return joblib.load(model_path)
    return None

@st.cache_data
def load_data(coin: str, data_type: str = "train"):
    """Veriyi yükle (cache'lenmiş)"""
    if data_type == "train":
        path = COINS[coin]["train"]
    else:
        path = COINS[coin]["test"]
    
    if path.exists():
        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["date"])
        return df
    return None

def get_live_data(symbol: str, days: int = 60):
    """Binance'den canlı veri çek"""
    try:
        client = Client()
        klines = client.get_klines(
            symbol=symbol,
            interval=Client.KLINE_INTERVAL_1DAY,
            limit=days
        )
        
        df = pd.DataFrame(klines, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume",
            "number_of_trades", "taker_buy_base",
            "taker_buy_quote", "ignore"
        ])
        
        df["date"] = pd.to_datetime(df["open_time"], unit="ms")
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)
        
        return df[["date", "open", "high", "low", "close", "volume"]].sort_values("date")
    except Exception as e:
        st.error(f"Veri çekme hatası: {e}")
        return None

def add_features_for_prediction(df: pd.DataFrame) -> pd.DataFrame:
    """Tahmin için feature engineering"""
    df = df.sort_values("date").copy()
    df["return"] = df["close"].pct_change()
    df["volatility"] = df["return"].rolling(window=14).std()
    df["range"] = df["high"] - df["low"]
    df["body"] = df["close"] - df["open"]
    df["return_lag_1"] = df["return"].shift(1)
    df["return_lag_3"] = df["return"].shift(3)
    df["ma_7"] = df["close"].rolling(7).mean()
    df["ma_7_diff"] = (df["close"] - df["ma_7"]) / df["ma_7"]
    df["range_pct"] = df["range"] / df["close"]
    return df.dropna()

# ======================================================
# SIDEBAR
# ======================================================
st.sidebar.title("⚙️ Ayarlar")

# Coin seçimi
available_coins = [coin for coin in COINS.keys() if COINS[coin]["model"].exists()]
if not available_coins:
    st.sidebar.error("⚠️ Hiçbir coin için model bulunamadı!")
    st.stop()

selected_coin = st.sidebar.selectbox(
    "Coin Seçin",
    available_coins,
    format_func=lambda x: f"{x} - {COINS[x]['name']}"
)

# Sayfa seçimi
page = st.sidebar.radio(
    "Sayfa",
    ["📊 Genel Bakış", "🔮 Canlı Tahmin", "📈 Veri Analizi", "📋 Model Performansı"]
)

# ======================================================
# ANA İÇERİK
# ======================================================

st.title("📊 Kripto Risk Tahmin Sistemi")
st.markdown(f"**Seçili Coin:** {selected_coin} - {COINS[selected_coin]['name']}")

# Model kontrolü
model = load_model(selected_coin)
if model is None:
    st.error(f"❌ {selected_coin} için model bulunamadı! Lütfen önce modeli eğitin.")
    st.stop()

# Feature listesi
FEATURES = [
    "close", "volume", "return", "body", "range",
    "return_lag_1", "return_lag_3", "ma_7_diff", "range_pct"
]

# ======================================================
# SAYFA İÇERİKLERİ
# ======================================================

if page == "📊 Genel Bakış":
    st.header("Genel Bakış")
    
    col1, col2, col3 = st.columns(3)
    
    # Veri yükle
    train_df = load_data(selected_coin, "train")
    test_df = load_data(selected_coin, "test")
    
    if train_df is not None and test_df is not None:
        # İstatistikler
        with col1:
            st.metric("Train Örnek Sayısı", f"{len(train_df):,}")
            st.metric("Test Örnek Sayısı", f"{len(test_df):,}")
        
        with col2:
            latest_train = train_df.iloc[-1]
            latest_test = test_df.iloc[-1] if len(test_df) > 0 else None
            if latest_test is not None:
                st.metric("Son Fiyat (Test)", f"${latest_test['close']:,.2f}")
            else:
                st.metric("Son Fiyat (Train)", f"${latest_train['close']:,.2f}")
        
        with col3:
            if "risk" in train_df.columns:
                risk_dist = train_df["risk"].value_counts().sort_index()
                most_common_risk = risk_dist.idxmax()
                risk_info = RISK_LABELS[most_common_risk]
                st.metric("En Yaygın Risk", f"{risk_info['color']} {risk_info['label']}")
        
        # Risk dağılımı grafiği
        if "risk" in train_df.columns:
            st.subheader("Risk Dağılımı (Train)")
            risk_counts = train_df["risk"].value_counts().sort_index()
            fig = px.bar(
                x=[RISK_LABELS[i]["label"] for i in risk_counts.index],
                y=risk_counts.values,
                color=[RISK_LABELS[i]["color_hex"] for i in risk_counts.index],
                labels={"x": "Risk Seviyesi", "y": "Örnek Sayısı"},
                title="Risk Sınıf Dağılımı"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Fiyat zaman serisi
        st.subheader("Fiyat Zaman Serisi")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=train_df["date"],
            y=train_df["close"],
            name="Train",
            line=dict(color="blue")
        ))
        if len(test_df) > 0:
            fig.add_trace(go.Scatter(
                x=test_df["date"],
                y=test_df["close"],
                name="Test",
                line=dict(color="red")
            ))
        fig.update_layout(
            title="Kapanış Fiyatı Zaman Serisi",
            xaxis_title="Tarih",
            yaxis_title="Fiyat (USDT)",
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)

elif page == "🔮 Canlı Tahmin":
    st.header("Canlı Risk Tahmini")
    
    # Binance'den veri çek
    if st.button("🔄 Güncel Veriyi Çek", type="primary"):
        with st.spinner("Binance'den veri çekiliyor..."):
            # 60 günlük tahmin için biraz daha fazla veri çekelim (indicatorler için)
            live_df = get_live_data(COINS[selected_coin]["symbol"], days=90)
            
            if live_df is not None:
                st.success(f"✅ {len(live_df)} günlük veri çekildi!")
                
                # Feature engineering
                # Tahmin için feature'ları ekle
                df_with_features = add_features_for_prediction(live_df)
                
                if len(df_with_features) > 0:
                    # Tüm veri için tahmin yap
                    # Feature engineering sonrası baştaki bazı satırlar NaN olup düşmüş olabilir
                    # O yüzden elimizdeki tüm veriye tahmin yapıyoruz
                    X_all = df_with_features[FEATURES]
                    all_preds = model.predict(X_all)
                    all_probs = model.predict_proba(X_all)
                    
                    # Tahminleri dataframe'e ekle
                    df_with_features["risk_pred"] = all_preds
                    
                    # Tabs
                    tab_today, tab_week, tab_month, tab_2months = st.tabs(["Bugün", "Son 7 Gün", "Son 1 Ay", "Son 60 Gün"])
                    
                    # --- TAB 1: BUGÜN ---
                    with tab_today:
                        # Son gün için tahmin
                        last_row = df_with_features.iloc[-1]
                        risk_pred = last_row["risk_pred"]
                        risk_proba = all_probs[-1]
                        
                        # Sonuçları göster
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Tarih", last_row["date"].strftime("%Y-%m-%d"))
                            st.metric("Fiyat", f"${last_row['close']:,.2f}")
                        
                        with col2:
                            risk_info = RISK_LABELS[risk_pred]
                            st.metric(
                                "Tahmin Edilen Risk",
                                f"{risk_info['color']} {risk_info['label']}",
                                delta=f"Risk Seviyesi: {risk_pred}"
                            )
                        
                        with col3:
                            st.metric("Volatilite", f"{last_row['volatility']:.6f}")
                            st.metric("Günlük Getiri", f"{last_row['return']*100:.2f}%")
                        
                        # Olasılık dağılımı
                        st.subheader("Risk Olasılıkları")
                        proba_df = pd.DataFrame({
                            "Risk Seviyesi": [RISK_LABELS[i]["label"] for i in range(3)],
                            "Olasılık": risk_proba
                        })
                        fig = px.bar(
                            proba_df,
                            x="Risk Seviyesi",
                            y="Olasılık",
                            color="Risk Seviyesi",
                            color_discrete_map={
                                RISK_LABELS[0]["label"]: RISK_LABELS[0]["color_hex"],
                                RISK_LABELS[1]["label"]: RISK_LABELS[1]["color_hex"],
                                RISK_LABELS[2]["label"]: RISK_LABELS[2]["color_hex"]
                            },
                            title="Risk Sınıfı Olasılık Dağılımı"
                        )
                        fig.update_layout(yaxis_tickformat=".2%")
                        st.plotly_chart(fig, use_container_width=True)

                    # --- YARDIMCI GÖRSELLEŞTİRME FONKSİYONU ---
                    def create_risk_history_chart(data, title):
                        fig = go.Figure()
                        
                        # Fiyat Çizgisi
                        fig.add_trace(go.Scatter(
                            x=data["date"],
                            y=data["close"],
                            name="Fiyat",
                            mode="lines",
                            line=dict(color="gray", width=1)
                        ))
                        
                        # Risk Noktaları
                        for risk_val in RISK_LABELS:
                            mask = data["risk_pred"] == risk_val
                            if mask.any():
                                fig.add_trace(go.Scatter(
                                    x=data[mask]["date"],
                                    y=data[mask]["close"],
                                    name=RISK_LABELS[risk_val]["label"],
                                    mode="markers",
                                    marker=dict(
                                        color=RISK_LABELS[risk_val]["color_hex"],
                                        size=10,
                                        symbol="circle"
                                    )
                                ))
                        
                        fig.update_layout(
                            title=title,
                            xaxis_title="Tarih",
                            yaxis_title="Fiyat (USDT)",
                            hovermode="x unified"
                        )
                        return fig

                    def create_risk_table(data):
                         # Tablo için veri hazırlığı
                        table_df = data[["date", "close", "risk_pred"]].copy()
                        table_df["date"] = table_df["date"].dt.strftime("%Y-%m-%d")
                        table_df["Risk"] = table_df["risk_pred"].apply(lambda x: f"{RISK_LABELS[x]['color']} {RISK_LABELS[x]['label']}")
                        table_df = table_df.rename(columns={"date": "Tarih", "close": "Fiyat", "risk_pred": "Risk Kodu"})
                        return table_df[["Tarih", "Fiyat", "Risk"]]

                    # --- TAB 2: SON 7 GÜN ---
                    with tab_week:
                        st.subheader("Son 7 Günlük Risk Analizi")
                        last_7_days = df_with_features.tail(7).sort_values("date", ascending=False)
                        st.dataframe(create_risk_table(last_7_days), use_container_width=True, hide_index=True)
                        last_7_days_chron = last_7_days.sort_values("date")
                        st.plotly_chart(create_risk_history_chart(last_7_days_chron, "Son 7 Gün Fiyat ve Risk"), use_container_width=True)

                    # --- TAB 3: SON 1 AY ---
                    with tab_month:
                        st.subheader("Son 30 Günlük Risk Analizi")
                        last_30_days = df_with_features.tail(30).sort_values("date", ascending=False)
                        st.dataframe(create_risk_table(last_30_days), use_container_width=True, hide_index=True)
                        last_30_days_chron = last_30_days.sort_values("date")
                        st.plotly_chart(create_risk_history_chart(last_30_days_chron, "Son 30 Gün Fiyat ve Risk"), use_container_width=True)

                    # --- TAB 4: SON 60 GÜN ---
                    with tab_2months:
                        st.subheader("Son 60 Günlük Risk Analizi")
                        last_60_days = df_with_features.tail(60).sort_values("date", ascending=False)
                        st.dataframe(create_risk_table(last_60_days), use_container_width=True, hide_index=True)
                        last_60_days_chron = last_60_days.sort_values("date")
                        st.plotly_chart(create_risk_history_chart(last_60_days_chron, "Son 60 Gün Fiyat ve Risk"), use_container_width=True)
                        
                else:
                    st.warning("Yeterli veri yok!")
            else:
                st.error("Veri çekilemedi!")

elif page == "📈 Veri Analizi":
    st.header("Veri Analizi")
    
    train_df = load_data(selected_coin, "train")
    test_df = load_data(selected_coin, "test")
    
    if train_df is not None:
        # Korelasyon matrisi
        st.subheader("Korelasyon Matrisi")
        numeric_cols = train_df.select_dtypes(include=[np.number]).columns
        corr_matrix = train_df[numeric_cols].corr()
        fig = px.imshow(
            corr_matrix,
            labels=dict(x="Değişken", y="Değişken", color="Korelasyon"),
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            color_continuous_scale="RdBu",
            aspect="auto",
            text_auto=".2f"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature dağılımları
        st.subheader("Feature Dağılımları")
        feature_to_plot = st.selectbox(
            "Görselleştirilecek Feature",
            ["return", "volatility", "range", "body", "volume"]
        )
        if feature_to_plot in train_df.columns:
            fig = px.histogram(
                train_df,
                x=feature_to_plot,
                nbins=50,
                title=f"{feature_to_plot.upper()} Dağılımı"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Feature importance
        if hasattr(model, "feature_importances_"):
            st.subheader("Feature Importance")
            importance_df = pd.DataFrame({
                "Feature": FEATURES,
                "Importance": model.feature_importances_
            }).sort_values("Importance", ascending=False)
            
            fig = px.bar(
                importance_df,
                x="Importance",
                y="Feature",
                orientation="h",
                title="Feature Importance (Random Forest)",
                labels={"Importance": "Önem Skoru", "Feature": "Feature"}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.dataframe(importance_df, use_container_width=True)

elif page == "📋 Model Performansı":
    st.header("Model Performansı")
    
    test_df = load_data(selected_coin, "test")
    
    if test_df is not None and "risk" in test_df.columns:
        # Test seti tahminleri
        X_test = test_df[FEATURES]
        y_test = test_df["risk"]
        y_pred = model.predict(X_test)
        
        # Accuracy
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
        accuracy = accuracy_score(y_test, y_pred)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Test Accuracy", f"{accuracy:.4f}")
        
        # Confusion Matrix
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        cm_df = pd.DataFrame(
            cm,
            index=[RISK_LABELS[i]["label"] for i in range(3)],
            columns=[RISK_LABELS[i]["label"] for i in range(3)]
        )
        fig = px.imshow(
            cm_df,
            labels=dict(x="Tahmin", y="Gerçek", color="Sayı"),
            x=cm_df.columns,
            y=cm_df.index,
            color_continuous_scale="Blues",
            aspect="auto",
            text_auto=True
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Classification Report
        st.subheader("Classification Report")
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df, use_container_width=True)
        
        # Zaman serisi tahmin karşılaştırması
        st.subheader("Gerçek vs Tahmin (Zaman Serisi)")
        comparison_df = pd.DataFrame({
            "date": test_df["date"],
            "Gerçek": y_test,
            "Tahmin": y_pred
        })
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=comparison_df["date"],
            y=comparison_df["Gerçek"],
            name="Gerçek",
            mode="lines+markers",
            line=dict(color="blue")
        ))
        fig.add_trace(go.Scatter(
            x=comparison_df["date"],
            y=comparison_df["Tahmin"],
            name="Tahmin",
            mode="lines+markers",
            line=dict(color="red", dash="dash")
        ))
        fig.update_layout(
            title="Gerçek vs Tahmin Risk Seviyeleri",
            xaxis_title="Tarih",
            yaxis_title="Risk Seviyesi",
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Test verisi bulunamadı veya risk etiketi yok!")

# Footer
st.markdown("---")
st.markdown("**⚠️ Uyarı:** Bu proje sadece eğitim amaçlıdır. Yatırım tavsiyesi değildir.")

