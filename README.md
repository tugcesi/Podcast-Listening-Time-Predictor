# 🎙️ Podcast Listening Time Predictor

> **Kaggle Playground Series S5E4** yarışması için geliştirilen makine öğrenmesi projesi.  
> Bir podcast bölümünün tahmini dinlenme süresini (dakika) XGBoost regresyon modeliyle tahmin eder.

---

## 📌 Proje Hakkında

Bu proje, podcast bölümlerine ait çeşitli özellikleri (bölüm uzunluğu, host popülerliği, yayın zamanı vb.) kullanarak dinlenme süresini tahmin eden uçtan uca bir ML pipeline'ı içermektedir. Tahmin sonuçları interaktif bir **Streamlit** uygulaması üzerinden sunulmaktadır.

| | |
|---|---|
| **Görev** | Regression |
| **Hedef Değişken** | `Listening_Time_minutes` |
| **Model** | XGBRegressor |
| **R²** | 0.7676 |
| **RMSE** | 13.08 dk |

---

## 📁 Dosya Yapısı

```
├── train.csv                    # Eğitim verisi (Kaggle'dan indirilmeli)
├── test.csv                     # Test verisi (Kaggle'dan indirilmeli)
├── save_model.py                # Model eğitimi ve artifact kaydetme
├── app.py                       # Streamlit uygulaması
├── requirements.txt             # Gerekli kütüphaneler
├── model.joblib                 # Eğitilmiş model (save_model.py çalıştırınca oluşur)
├── features.joblib              # Özellik listesi
├── impute_artifacts.joblib      # Eksik değer doldurma artifact'ları
└── encoding_artifacts.joblib    # Encoding artifact'ları
```

---

## 🔧 Kurulum

### 1. Repoyu Klonlayın

```bash
git clone https://github.com/tugcesi/Podcast-Listening-Time-Predictor.git
cd Podcast-Listening-Time-Predictor
```

### 2. Bağımlılıkları Yükleyin

```bash
pip install -r requirements.txt
```

### 3. Veri Setini İndirin

Veri setini [Kaggle](https://www.kaggle.com/competitions/playground-series-s5e4) üzerinden indirip proje kök dizinine yerleştirin:

```
train.csv
test.csv
```

---

## 🚀 Kullanım

### Adım 1 — Modeli Eğitin

```bash
python save_model.py
```

Bu komut şu dosyaları oluşturur:
- `model.joblib`
- `features.joblib`
- `impute_artifacts.joblib`
- `encoding_artifacts.joblib`

### Adım 2 — Uygulamayı Başlatın

```bash
streamlit run app.py
```

Tarayıcınızda `http://localhost:8501` adresini açın.

---

## ⚙️ Pipeline Detayları

### Eksik Değer Doldurma

| Sütun | Strateji |
|---|---|
| `Episode_Length_minutes` | Genre + Podcast_Name → Genre → Overall Median |
| `Guest_Popularity_percentage` | Podcast_Name → Genre → Overall Median |
| `Number_of_Ads` | Mode |

### Feature Engineering

| Özellik | Formül |
|---|---|
| `length_per_ad` | `Episode_Length_minutes / (Number_of_Ads + 1)` |

### Encoding

| Sütun | Yöntem | Detay |
|---|---|---|
| `Episode_Sentiment` | Ordinal | Negative→0, Neutral→1, Positive→2 |
| `Publication_Time` | Ordinal | Morning→1, Afternoon→2, Evening→3, Night→4 |
| `Genre` | Target Encoding | Train seti `Listening_Time_minutes` ortalaması |
| `Publication_Day` | Target Encoding | Train seti `Listening_Time_minutes` ortalaması |
| `Podcast_Name` | Target Encoding | Train seti `Listening_Time_minutes` ortalaması |

### Kullanılan Özellikler

```
Episode_Length_minutes
length_per_ad
Number_of_Ads
Host_Popularity_percentage
Episode_Sentiment
Publication_Time
Genre
Publication_Day
Podcast_Name
```

---

## 📊 Model Performansı

| Metrik | Değer |
|---|---|
| R�� | 0.7676 |
| RMSE | 13.08 dk |
| MAE | — |

---

## 🖥️ Uygulama Ekran Görüntüsü

Sidebar'dan episode bilgilerini doldurduktan sonra **Tahmin Et** butonuna tıklayın.  
Sonuçlar gauge grafiği, metrik kartları ve feature importance grafiği ile görüntülenir.

---

## 📚 Veri Seti

- **Kaynak:** [Kaggle — Playground Series S5E4](https://www.kaggle.com/competitions/playground-series-s5e4)
- **Eğitim seti boyutu:** ~50.000 satır
- **Özellik sayısı:** 9 (ham) → 10 (feature engineering sonrası)

---

## 🛠️ Kullanılan Teknolojiler

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-2.x-orange?logo=xgboost)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red?logo=streamlit)
![Pandas](https://img.shields.io/badge/Pandas-2.x-150458?logo=pandas)
![scikit--learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?logo=scikitlearn)

---

## 📄 Lisans

Bu proje [MIT Lisansı](LICENSE) kapsamında lisanslanmıştır.
