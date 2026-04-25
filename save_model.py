import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import joblib
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# ── 1. Veri Yükleme ───────────────────────────────────────────────────────────
print("📥 Veri yükleniyor...")
train = pd.read_csv("train.csv")

# ── 2. Eksik Değer İmputation (train üzerinden) ──────────────────────────────

# Episode_Length_minutes: Genre+Podcast_Name → Genre → overall median
m1 = train.groupby(['Genre', 'Podcast_Name'])['Episode_Length_minutes'].median().reset_index()
m2 = train.groupby('Genre')['Episode_Length_minutes'].median().reset_index()
m3 = train['Episode_Length_minutes'].median()

train = train.merge(m1, on=['Genre', 'Podcast_Name'], how='left', suffixes=('', '_m1'))
train['Episode_Length_minutes'] = train['Episode_Length_minutes'].fillna(train['Episode_Length_minutes_m1'])
train.drop(columns='Episode_Length_minutes_m1', inplace=True)

train = train.merge(m2, on='Genre', how='left', suffixes=('', '_m2'))
train['Episode_Length_minutes'] = train['Episode_Length_minutes'].fillna(train['Episode_Length_minutes_m2'])
train.drop(columns='Episode_Length_minutes_m2', inplace=True)

train['Episode_Length_minutes'] = train['Episode_Length_minutes'].fillna(m3)

# Guest_Popularity_percentage: Podcast_Name → Genre → overall median
gp_podcast = train.groupby('Podcast_Name')['Guest_Popularity_percentage'].median()
gp_genre   = train.groupby('Genre')['Guest_Popularity_percentage'].median()
gp_overall = train['Guest_Popularity_percentage'].median()

train['Guest_Popularity_percentage'] = (
    train['Guest_Popularity_percentage']
    .fillna(train['Podcast_Name'].map(gp_podcast))
    .fillna(train['Genre'].map(gp_genre))
    .fillna(gp_overall)
)

# Number_of_Ads: mode
ads_mode = train['Number_of_Ads'].mode()[0]
train['Number_of_Ads'] = train['Number_of_Ads'].fillna(ads_mode)

# ── 3. Feature Engineering ──────────��────────────────────────────────────────
train['length_per_ad']  = train['Episode_Length_minutes'] / (train['Number_of_Ads'] + 1)
train['avg_popularity'] = (train['Host_Popularity_percentage'] + train['Guest_Popularity_percentage']) / 2

# ── 4. Encoding ──────────────────────────────────────────────────────────────

# Ordinal
sentiment_map = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
time_map      = {'Morning': 1, 'Afternoon': 2, 'Evening': 3, 'Night': 4}

train['Episode_Sentiment'] = train['Episode_Sentiment'].map(sentiment_map)
train['Publication_Time']  = train['Publication_Time'].map(time_map)

# Target encoding (train mean)
genre_mean  = train.groupby('Genre')['Listening_Time_minutes'].mean()
day_mean    = train.groupby('Publication_Day')['Listening_Time_minutes'].mean()
podcast_mean= train.groupby('Podcast_Name')['Listening_Time_minutes'].mean()
global_mean = train['Listening_Time_minutes'].mean()

train['Genre']           = train['Genre'].map(genre_mean)
train['Publication_Day'] = train['Publication_Day'].map(day_mean)
train['Podcast_Name']    = train['Podcast_Name'].map(podcast_mean)

# ── 5. X / y ─────────────────────────────────────────────────────────────────
features = [
    'Episode_Length_minutes',
    'length_per_ad',
    'Number_of_Ads',
    'Host_Popularity_percentage',
    'Episode_Sentiment',
    'Publication_Time',
    'Genre',
    'Publication_Day',
    'Podcast_Name'
]

X = train[features]
y = train['Listening_Time_minutes']

# ── 6. Değerlendirme (80/20 split) ───────────────────────────────────────────
X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
eval_model = XGBRegressor(random_state=42)
eval_model.fit(X_tr, y_tr)
preds = eval_model.predict(X_val)

print(f"\n📊 Validation Sonuçları:")
print(f"   R²  : {r2_score(y_val, preds):.4f}")
print(f"   RMSE: {mean_squared_error(y_val, preds)**0.5:.4f}")
print(f"   MAE : {mean_absolute_error(y_val, preds):.4f}")

# ── 7. Full train üzerinde final model ───────────────────────────────────────
print("\n🚀 Final model eğitiliyor (full train)...")
best_model = XGBRegressor(random_state=42)
best_model.fit(X, y)

# ── 8. Artifact Kaydetme ─────────────────────────────────────────────────────
impute_artifacts = {
    'episode_length_m1': m1,   # DataFrame
    'episode_length_m2': m2,   # DataFrame
    'episode_length_m3': m3,   # scalar
    'guest_pop_podcast': gp_podcast,
    'guest_pop_genre':   gp_genre,
    'guest_pop_overall': gp_overall,
    'ads_mode':          ads_mode,
}

encoding_artifacts = {
    'sentiment_map':  sentiment_map,
    'time_map':       time_map,
    'genre_mean':     genre_mean,
    'day_mean':       day_mean,
    'podcast_mean':   podcast_mean,
    'global_mean':    global_mean,
}

joblib.dump(best_model,         'model.joblib')
joblib.dump(features,           'features.joblib')
joblib.dump(impute_artifacts,   'impute_artifacts.joblib')
joblib.dump(encoding_artifacts, 'encoding_artifacts.joblib')

print("\n✅ Kaydedilen dosyalar:")
print("   model.joblib")
print("   features.joblib")
print("   impute_artifacts.joblib")
print("   encoding_artifacts.joblib")