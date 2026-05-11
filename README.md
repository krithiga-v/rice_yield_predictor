# 🌾 Rice Yield Predictor — Tamil Nadu

> Predicts district-level rice yield using **real satellite imagery**, **NASA weather data**, and a **machine learning ensemble** (Random Forest + XGBoost).

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)
![Google Earth Engine](https://img.shields.io/badge/Google-Earth%20Engine-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📸 Dashboard Preview

- 🗺️ Interactive Tamil Nadu yield map
- 📈 14-year historical trend per district
- 📊 2023 vs 2024 forecast bar chart
- 📋 District ranking table
- 🎯 Live yield predictor with 4 input sliders

---

## 🛰️ Data Sources

| Source | What it provides |
|---|---|
| **Sentinel-2 / Landsat-8** (ESA/USGS) | NDVI (crop health index) via Google Earth Engine |
| **NASA POWER API** | Daily temperature & rainfall (2013–2023) |
| **TN Agriculture Dept** | Historical rice yield ground truth |

---

## 🤖 Models Trained

| Model | MAE | Result |
|---|---|---|
| Random Forest | 0.309 t/ha | 🥇 Winner |
| XGBoost | 0.315 t/ha | 🥈 Runner-up |
| LSTM (Deep Learning) | 1.049 t/ha | Needs more data |
| **RF + XGB Ensemble** | **Best predictions** | 🏆 Used in dashboard |

> RF vs XGBoost correlation: **0.952** — high confidence predictions ✅

---

## 📊 Dataset

| Property | Value |
|---|---|
| Districts | 15 Tamil Nadu rice-growing districts |
| Years | 2013 – 2023 (11 years) |
| Training rows | 165 |
| Features | 6 (NDVI, Temp, Rain, etc.) |

### Districts covered
Thanjavur · Tiruvarur · Trichy · Villupuram · Nagapattinam · Ariyalur · Pudukkottai · Cuddalore · Madurai · Perambalur · Erode · Salem · Dindigul · Namakkal · Dharmapuri

### Features used
| Feature | Description |
|---|---|
| `Mean_NDVI` | Crop health index from satellite imagery |
| `Avg_Temp_C` | Average seasonal temperature |
| `Max_Temp_C` | Maximum temperature during season |
| `Min_Temp_C` | Minimum temperature during season |
| `Total_Rain_mm` | Total seasonal rainfall |
| `Rainy_Days` | Number of rainy days in season |

---

## 🚀 How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/rice-yield-predictor.git
cd rice-yield-predictor
```

### 2. Create virtual environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac / Linux
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Authenticate Google Earth Engine
```bash
python -c "import ee; ee.Authenticate()"
```

### 5. Run the dashboard
```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501` 🎉

---

## 📁 Project Structure

```
rice-yield-predictor/
│
├── app.py                  ← Streamlit dashboard (main file)
├── notebook.ipynb          ← Full development notebook (Google Colab)
├── requirements.txt        ← Python package dependencies
├── .gitignore              ← Files to exclude from Git
└── README.md               ← This file
```

---

## 🧠 Key Learnings

- **More data = better model** → MAE improved from 0.436 → 0.309 t/ha
- **Random Forest beats LSTM** on small tabular datasets (165 rows)
- **Ensemble models** are more reliable than single models
- **NDVI from satellite** is the strongest predictor of rice yield
- **Temperature** is the second most important feature

---

## 🔮 2024 Predictions — Top 5 Districts

| Rank | District | 2023 Actual | 2024 Forecast | Change |
|---|---|---|---|---|
| 1 | Thanjavur | 4.30 t/ha | 4.14 t/ha | ↓ 0.16 |
| 2 | Tiruvarur | 4.00 t/ha | 3.91 t/ha | ↓ 0.09 |
| 3 | Trichy | 3.50 t/ha | 3.24 t/ha | ↓ 0.26 |
| 4 | Villupuram | 3.20 t/ha | 3.24 t/ha | ↑ 0.04 |
| 5 | Nagapattinam | 3.20 t/ha | 3.18 t/ha | ↓ 0.02 |

**TN Average 2024 Forecast: 3.17 t/ha**

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Satellite data | Google Earth Engine, `earthengine-api`, `geemap` |
| Weather data | NASA POWER REST API |
| Data processing | `pandas`, `numpy` |
| Machine Learning | `scikit-learn`, `xgboost`, `tensorflow` |
| Visualization | `matplotlib`, `seaborn`, `folium` |
| Dashboard | `streamlit` |

---

## 📈 Model Evolution

```
V1 — 5 rows   (1 year,  5 districts)  → MAE: 0.436 t/ha
V2 — 25 rows  (5 years, 5 districts)  → MAE: 0.380 t/ha
V3 — 75 rows  (5 years, 15 districts) → MAE: 0.295 t/ha
V4 — 165 rows (11 years,15 districts) → MAE: 0.309 t/ha ✅
```

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

*Built as a data science portfolio project — Tamil Nadu rice yield prediction using real geospatial and weather data.*
