import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
import os

# ────────────────────────────────────────────────────────��────────────────────
# Sayfa Ayarları
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🎙️ Podcast Listening Time Predictor",
    page_icon="🎙️",
    layout="wide"
)

st.title("🎙️ Podcast Listening Time Predictor")
st.markdown("Bir podcast bölümünün **tahmini dinlenme süresini** tahmin edin.")
st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# Artifact Yükleme
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    required = ['model.joblib', 'features.joblib',
                'impute_artifacts.joblib', 'encoding_artifacts.joblib']
    missing = [f for f in required if not os.path.exists(f)]
    if missing:
        return None, None, None, None, f"Eksik dosyalar: {missing}"
    model    = joblib.load('model.joblib')
    features = joblib.load('features.joblib')
    impute   = joblib.load('impute_artifacts.joblib')
    encoding = joblib.load('encoding_artifacts.joblib')
    return model, features, impute, encoding, None

model, features, impute, encoding, err = load_artifacts()

if err:
    st.error(f"⚠️ Model dosyaları bulunamadı!\n\n`{err}`")
    st.info("Önce şu komutu çalıştırın: `python save_model.py`")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# Bilinen Kategoriler (encoding artifact'larından)
# ─────────────────────────────────────────────────────────────────────────────
known_genres   = sorted(encoding['genre_mean'].index.tolist())
known_days     = sorted(encoding['day_mean'].index.tolist())
known_podcasts = sorted(encoding['podcast_mean'].index.tolist())

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar: Kullanıcı Girdileri
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🎛️ Episode Bilgileri")

    st.subheader("📻 Podcast & Bölüm")
    podcast_name = st.selectbox("Podcast Adı", known_podcasts)
    genre        = st.selectbox("Tür (Genre)", known_genres)

    st.subheader("⏱️ Bölüm Özellikleri")
    episode_length = st.slider(
        "Bölüm Uzunluğu (dakika)", min_value=1.0, max_value=200.0, value=60.0, step=0.5
    )
    number_of_ads  = st.slider("Reklam Sayısı", min_value=0, max_value=20, value=2)

    st.subheader("👥 Popülerlik")
    host_popularity  = st.slider(
        "Host Popülerlik (%)", min_value=0.0, max_value=120.0, value=65.0, step=0.1
    )
    guest_popularity = st.slider(
        "Konuk Popülerlik (%)", min_value=0.0, max_value=120.0, value=50.0, step=0.1
    )

    st.subheader("📅 Yayın Bilgisi")
    publication_day  = st.selectbox("Yayın Günü", known_days)
    publication_time = st.selectbox(
        "Yayın Zamanı", ["Morning", "Afternoon", "Evening", "Night"]
    )

    st.subheader("😊 Bölüm Tonu")
    episode_sentiment = st.selectbox(
        "Episode Sentiment", ["Positive", "Neutral", "Negative"]
    )

    predict_btn = st.button("🔮 Tahmin Et", use_container_width=True, type="primary")

# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing & Tahmin
# ─────────────────────────────────────────────────────────────────────────────
def preprocess_input(inputs: dict, impute: dict, encoding: dict) -> pd.DataFrame:
    # Feature engineering
    length_per_ad  = inputs['Episode_Length_minutes'] / (inputs['Number_of_Ads'] + 1)
    avg_popularity = (inputs['Host_Popularity_percentage'] + inputs['Guest_Popularity_percentage']) / 2  # noqa

    # Ordinal encoding
    sentiment_val = encoding['sentiment_map'].get(inputs['Episode_Sentiment'], 1)
    time_val      = encoding['time_map'].get(inputs['Publication_Time'], 2)

    # Target encoding (unknown → global mean)
    gm = encoding['global_mean']
    genre_val   = encoding['genre_mean'].get(inputs['Genre'],           gm)
    day_val     = encoding['day_mean'].get(inputs['Publication_Day'],   gm)
    podcast_val = encoding['podcast_mean'].get(inputs['Podcast_Name'],  gm)

    row = {
        'Episode_Length_minutes':   inputs['Episode_Length_minutes'],
        'length_per_ad':            length_per_ad,
        'Number_of_Ads':            inputs['Number_of_Ads'],
        'Host_Popularity_percentage': inputs['Host_Popularity_percentage'],
        'Episode_Sentiment':        sentiment_val,
        'Publication_Time':         time_val,
        'Genre':                    genre_val,
        'Publication_Day':          day_val,
        'Podcast_Name':             podcast_val,
    }
    return pd.DataFrame([row])


def make_gauge(value: float, max_val: float = 120.0) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        delta={'reference': 45.4, 'valueformat': '.1f'},
        title={'text': "Tahmini Dinlenme Süresi (dakika)", 'font': {'size': 16}},
        number={'suffix': " dk", 'valueformat': '.1f'},
        gauge={
            'axis': {'range': [0, max_val], 'tickwidth': 1},
            'bar':  {'color': "#1DB954"},
            'steps': [
                {'range': [0,    30],  'color': '#ffeaa7'},
                {'range': [30,   70],  'color': '#b2dfdb'},
                {'range': [70, max_val], 'color': '#c8e6c9'},
            ],
            'threshold': {
                'line': {'color': 'red', 'width': 4},
                'thickness': 0.75,
                'value': 45.4,  # eğitim seti ortalaması
            }
        }
    ))
    fig.update_layout(height=280, margin=dict(t=30, b=10, l=20, r=20))
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Ana Ekran
# ─────────────────────────────────────────────────────────────────────────────
if predict_btn:
    inputs = {
        'Podcast_Name':               podcast_name,
        'Genre':                      genre,
        'Episode_Length_minutes':     episode_length,
        'Number_of_Ads':              float(number_of_ads),
        'Host_Popularity_percentage': host_popularity,
        'Guest_Popularity_percentage':guest_popularity,
        'Publication_Day':            publication_day,
        'Publication_Time':           publication_time,
        'Episode_Sentiment':          episode_sentiment,
    }

    try:
        X_input    = preprocess_input(inputs, impute, encoding)
        prediction = float(model.predict(X_input)[0])
        prediction = max(0.0, prediction)   # negatif olamaz

        # ── Sonuç kutusu ──────────────────────────────────────────────────
        col1, col2 = st.columns([1, 1])

        with col1:
            st.success(f"### ✅ Tahmini Dinlenme Süresi")
            st.metric(
                label="",
                value=f"{prediction:.1f} dakika",
                delta=f"{prediction - 45.4:+.1f} dk (ortalamaya göre)"
            )
            st.caption("🔴 Kırmızı çizgi = eğitim seti ortalaması (45.4 dk)")
            st.plotly_chart(make_gauge(prediction), use_container_width=True)

        with col2:
            st.info("### 📋 Girilen Bilgiler")
            display_df = pd.DataFrame({
                'Özellik': [
                    'Podcast Adı', 'Tür', 'Bölüm Uzunluğu',
                    'Reklam Sayısı', 'Host Popülerlik', 'Konuk Popülerlik',
                    'Yayın Günü', 'Yayın Zamanı', 'Sentiment'
                ],
                'Değer': [
                    podcast_name, genre, f"{episode_length} dk",
                    number_of_ads, f"%{host_popularity}", f"%{guest_popularity}",
                    publication_day, publication_time, episode_sentiment
                ]
            })
            st.dataframe(display_df, hide_index=True, use_container_width=True)

        # ── Feature Importance ────────────────────────────────────────────
        st.divider()
        st.subheader("📈 Feature Importance")

        feat_imp = pd.Series(
            model.feature_importances_,
            index=features
        ).sort_values(ascending=True)

        fig_imp = go.Figure(go.Bar(
            x=feat_imp.values,
            y=feat_imp.index,
            orientation='h',
            marker_color='#1DB954'
        ))
        fig_imp.update_layout(
            title="XGBoost Feature Importance",
            xaxis_title="Importance",
            yaxis_title="Feature",
            height=350,
            margin=dict(l=10, r=10, t=40, b=10)
        )
        st.plotly_chart(fig_imp, use_container_width=True)

    except Exception as e:
        st.error(f"⚠️ Tahmin sırasında hata: {e}")

else:
    # ── Karşılama ekranı ──────────────────────────────────────────────────────
    st.info("👈 Sol panelden bölüm bilgilerini doldurun ve **Tahmin Et** butonuna tıklayın.")

    st.subheader("ℹ️ Model Hakkında")
    col1, col2, col3 = st.columns(3)
    col1.metric("Model",  "XGBRegressor")
    col2.metric("R²",     "0.7676")
    col3.metric("RMSE",   "13.08 dk")

    st.subheader("🔬 Kullanılan Özellikler")
    feat_df = pd.DataFrame({
        'Özellik': [
            'Episode_Length_minutes', 'length_per_ad', 'Number_of_Ads',
            'Host_Popularity_percentage', 'Episode_Sentiment', 'Publication_Time',
            'Genre', 'Publication_Day', 'Podcast_Name'
        ],
        'Açıklama': [
            'Bölüm uzunluğu (dk)', 'Uzunluk / (reklam+1)',
            'Reklam sayısı', 'Host popülerlik yüzdesi',
            'Bölüm tonu (Ordinal)', 'Yayın zamanı (Ordinal)',
            'Tür (Target Encoding)', 'Yayın günü (Target Encoding)',
            'Podcast adı (Target Encoding)'
        ],
        'Tip': [
            'Sayısal', 'Türetilmiş', 'Sayısal',
            'Sayısal', 'Ordinal', 'Ordinal',
            'Target Enc.', 'Target Enc.', 'Target Enc.'
        ]
    })
    st.dataframe(feat_df, hide_index=True, use_container_width=True)