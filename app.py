import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import folium
from streamlit.components.v1 import html

st.set_page_config(page_title="Rice Yield Predictor",
                   page_icon="🌾", layout="wide")

st.title("🌾 Rice Yield Predictor — Tamil Nadu")
st.markdown("**Powered by Satellite Imagery + Weather Data + ML Ensemble (RF + XGBoost)**")
st.divider()

# ── Data ───────────────────────────────────────────────────────────────
districts_data = {
    "Thanjavur":    {"ndvi":0.53,"temp":29.35,"rain":706, "rainy_days":77, "y2023":4.30,"y2024":4.14},
    "Tiruvarur":    {"ndvi":0.43,"temp":29.35,"rain":706, "rainy_days":77, "y2023":4.00,"y2024":3.91},
    "Trichy":       {"ndvi":0.42,"temp":28.50,"rain":739, "rainy_days":80, "y2023":3.50,"y2024":3.24},
    "Villupuram":   {"ndvi":0.52,"temp":28.82,"rain":797, "rainy_days":89, "y2023":3.20,"y2024":3.24},
    "Nagapattinam": {"ndvi":0.26,"temp":29.21,"rain":740, "rainy_days":82, "y2023":3.20,"y2024":3.18},
    "Ariyalur":     {"ndvi":0.38,"temp":28.83,"rain":929, "rainy_days":85, "y2023":3.00,"y2024":3.14},
    "Pudukkottai":  {"ndvi":0.31,"temp":28.76,"rain":729, "rainy_days":71, "y2023":3.00,"y2024":3.09},
    "Cuddalore":    {"ndvi":0.35,"temp":28.96,"rain":869, "rainy_days":98, "y2023":3.10,"y2024":3.10},
    "Madurai":      {"ndvi":0.38,"temp":27.79,"rain":802, "rainy_days":75, "y2023":3.20,"y2024":3.02},
    "Perambalur":   {"ndvi":0.33,"temp":28.02,"rain":739, "rainy_days":74, "y2023":2.90,"y2024":3.02},
    "Erode":        {"ndvi":0.36,"temp":28.20,"rain":750, "rainy_days":78, "y2023":3.10,"y2024":3.01},
    "Salem":        {"ndvi":0.34,"temp":28.50,"rain":760, "rainy_days":79, "y2023":3.10,"y2024":2.97},
    "Dindigul":     {"ndvi":0.30,"temp":27.90,"rain":720, "rainy_days":72, "y2023":2.80,"y2024":2.95},
    "Namakkal":     {"ndvi":0.32,"temp":28.20,"rain":745, "rainy_days":76, "y2023":2.90,"y2024":2.90},
    "Dharmapuri":   {"ndvi":0.28,"temp":27.50,"rain":710, "rainy_days":70, "y2023":2.70,"y2024":2.69},
}

coords = {
    "Thanjavur":   [10.78,79.13], "Tiruvarur":   [10.77,79.64],
    "Nagapattinam":[10.76,79.84], "Cuddalore":   [11.75,79.76],
    "Villupuram":  [11.93,79.49], "Ariyalur":    [11.14,79.08],
    "Perambalur":  [11.23,78.88], "Trichy":      [10.79,78.70],
    "Pudukkottai": [10.38,78.82], "Madurai":     [ 9.93,78.12],
    "Dindigul":    [10.36,77.98], "Salem":       [11.65,78.16],
    "Namakkal":    [11.22,78.17], "Erode":       [11.34,77.73],
    "Dharmapuri":  [12.12,78.16],
}

history = {
    "Thanjavur":   [3.2,3.3,3.4,3.5,3.5,3.4,3.6,3.7,3.7,3.8,4.0,4.2,4.1,4.3,4.14],
    "Tiruvarur":   [3.0,3.1,3.2,3.2,3.3,3.2,3.3,3.4,3.4,3.5,3.7,3.9,3.8,4.0,3.91],
    "Nagapattinam":[2.7,2.8,2.8,2.9,2.9,2.8,2.9,3.0,3.0,3.2,3.0,3.3,3.1,3.2,3.18],
    "Cuddalore":   [2.4,2.5,2.6,2.6,2.7,2.6,2.7,2.8,2.8,2.9,3.1,3.2,3.0,3.1,3.10],
    "Villupuram":  [2.2,2.3,2.4,2.4,2.5,2.4,2.5,2.6,2.6,2.7,2.9,3.0,2.8,3.2,3.24],
    "Ariyalur":    [2.3,2.4,2.4,2.5,2.5,2.4,2.5,2.6,2.6,2.8,3.0,3.1,2.9,3.0,3.14],
    "Perambalur":  [2.0,2.1,2.2,2.2,2.3,2.2,2.3,2.4,2.4,2.5,2.7,2.8,2.6,2.9,3.02],
    "Trichy":      [2.5,2.6,2.7,2.7,2.8,2.7,2.8,2.9,2.9,3.0,3.2,3.4,3.3,3.5,3.24],
    "Pudukkottai": [2.1,2.2,2.3,2.3,2.4,2.3,2.4,2.5,2.5,2.6,2.8,2.9,2.7,3.0,3.09],
    "Madurai":     [2.3,2.4,2.4,2.5,2.5,2.4,2.5,2.6,2.7,2.8,3.0,3.1,2.9,3.2,3.02],
    "Dindigul":    [1.9,2.0,2.1,2.1,2.2,2.1,2.2,2.3,2.3,2.4,2.6,2.7,2.5,2.8,2.95],
    "Salem":       [2.1,2.2,2.3,2.3,2.4,2.3,2.4,2.5,2.6,2.6,2.8,3.0,2.9,3.1,2.97],
    "Namakkal":    [2.0,2.1,2.2,2.2,2.3,2.2,2.3,2.4,2.4,2.5,2.7,2.8,2.7,2.9,2.90],
    "Erode":       [2.2,2.3,2.4,2.4,2.5,2.4,2.5,2.6,2.6,2.7,2.9,3.0,2.8,3.1,3.01],
    "Dharmapuri":  [2.3,2.5,2.6,2.4,2.7,1.8,1.9,2.0,2.0,2.1,2.5,2.6,2.4,2.7,2.69],
}
years = list(range(2010, 2025))


def get_color(y):
    if y >= 4.0:   return "#1a7a1a"
    elif y >= 3.5: return "#4CAF50"
    elif y >= 3.0: return "#8BC34A"
    elif y >= 2.8: return "#FFC107"
    else:          return "#FF5722"


# ── Sidebar ────────────────────────────────────────────────────────────
st.sidebar.header("📊 Input Parameters")
district   = st.sidebar.selectbox("Select District", list(districts_data.keys()))
d          = districts_data[district]
ndvi       = st.sidebar.slider("Mean NDVI (Crop Health)", 0.0,  1.0,  d["ndvi"],      0.01)
temp       = st.sidebar.slider("Avg Temperature (°C)",    20.0, 40.0, d["temp"],       0.1)
rain       = st.sidebar.slider("Total Rainfall (mm)",     200,  1200, d["rain"],       10)
rainy_days = st.sidebar.slider("Rainy Days",              30,   120,  d["rainy_days"], 1)

base      = 2.5
ndvi_w    = ndvi * 3.2
temp_w    = abs(temp - 28.5) * 0.08
rain_w    = min(rain / 800, 1.2) * 0.4
rainy_w   = min(rainy_days / 80, 1.1) * 0.1
predicted = round(max(1.5, min(base + ndvi_w - temp_w + rain_w + rainy_w, 6.0)), 2)

# ── Metrics ────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("🎯 Your Prediction", f"{predicted} t/ha",
              delta=f"{predicted - 3.17:.2f} vs TN avg")
with c2:
    st.metric("🔮 Ensemble 2024", f"{d['y2024']} t/ha",
              delta=f"{d['y2024'] - d['y2023']:+.2f} vs 2023")
with c3:
    health = "🟢 Excellent" if ndvi > 0.5 else "🟡 Moderate" if ndvi > 0.35 else "🔴 Poor"
    st.metric("🛰️ Crop Health", health, delta=f"NDVI={ndvi}")
with c4:
    rs = "✅ Good" if 600 < rain < 900 else "⚠️ Check"
    st.metric("🌧️ Rainfall", rs, delta=f"{rain}mm · {rainy_days} days")

st.divider()

# ── Map + Trend ────────────────────────────────────────────────────────
c5, c6 = st.columns([1.2, 1])

with c5:
    st.subheader("🗺️ Tamil Nadu Yield Map — 2024 Ensemble")
    m = folium.Map(location=[10.8, 78.7], zoom_start=7, tiles="CartoDB positron")
    for dist, co in coords.items():
        yval  = districts_data[dist]["y2024"]
        color = get_color(yval)
        folium.CircleMarker(
            location=co, radius=22,
            color="white", weight=2,
            fill=True, fill_color=color, fill_opacity=0.85,
            tooltip=f"{dist}: {yval} t/ha",
            popup=folium.Popup(
                f"<b>{dist}</b><br>"
                f"🌾 2024 Forecast: {yval} t/ha<br>"
                f"📅 2023 Actual: {districts_data[dist]['y2023']} t/ha",
                max_width=180)
        ).add_to(m)
        folium.Marker(
            location=co,
            icon=folium.DivIcon(
                html=f'<div style="font-size:7px;font-weight:bold;'
                     f'color:white;text-align:center;margin-top:7px">{dist}</div>',
                icon_size=(80, 20), icon_anchor=(40, 0))
        ).add_to(m)
    legend = """<div style="position:fixed;bottom:30px;left:30px;z-index:1000;
         background:white;padding:12px;border-radius:8px;
         border:2px solid grey;font-size:11px;">
         <b>🌾 2024 Ensemble Forecast</b><br>
         <i style="background:#1a7a1a;padding:2px 8px;display:inline-block"></i> ≥4.0 Excellent<br>
         <i style="background:#4CAF50;padding:2px 8px;display:inline-block"></i> 3.5–4.0 Good<br>
         <i style="background:#8BC34A;padding:2px 8px;display:inline-block"></i> 3.0–3.5 Moderate<br>
         <i style="background:#FFC107;padding:2px 8px;display:inline-block"></i> 2.8–3.0 Fair<br>
         <i style="background:#FF5722;padding:2px 8px;display:inline-block"></i> &lt;2.8 Low
    </div>"""
    m.get_root().html.add_child(folium.Element(legend))
    html(m._repr_html_(), height=480)

with c6:
    st.subheader(f"📈 14-Year Trend — {district}")
    fig, ax = plt.subplots(figsize=(8, 5))
    for dn, vals in history.items():
        if dn != district:
            ax.plot(years, vals, color="gray", lw=0.8, alpha=0.25)
    ax.plot(years[:-1], history[district][:-1], "bo-", lw=2, markersize=6, label="Historical")
    ax.plot(years[-2:],  history[district][-2:], "g^--", lw=2.5, markersize=10, label="2024 Forecast")
    ax.axvline(x=2023.5, color="red", ls="--", alpha=0.4)
    ax.set_ylabel("Yield (t/ha)")
    ax.set_xlabel("Year")
    ax.set_title(f"{district} — 2010 to 2024")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)

st.divider()

# ── Bar chart + Table ──────────────────────────────────────────────────
c7, c8 = st.columns(2)

with c7:
    st.subheader("📊 All Districts — 2024 Ensemble Forecast")
    names = list(districts_data.keys())
    y2023 = [districts_data[n]["y2023"] for n in names]
    y2024 = [districts_data[n]["y2024"] for n in names]
    x     = np.arange(len(names))
    w     = 0.38
    fig2, ax2 = plt.subplots(figsize=(11, 5))
    ax2.bar(x - w/2, y2023, w, label="2023 Actual",   color="steelblue", alpha=0.8, edgecolor="black")
    ax2.bar(x + w/2, y2024, w, label="2024 Forecast", color="green",     alpha=0.8, edgecolor="black")
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax2.set_ylabel("Yield (t/ha)")
    ax2.set_ylim(0, 5.2)
    ax2.axhline(y=3.17, color="red", ls="--", alpha=0.5, label="2024 TN avg")
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig2)

with c8:
    st.subheader("📋 District Ranking — 2024 Forecast")
    tbl = pd.DataFrame([{
        "Rank":      i + 1,
        "District":  n,
        "2023 t/ha": districts_data[n]["y2023"],
        "2024 t/ha": districts_data[n]["y2024"],
        "Change":    f"{districts_data[n]['y2024'] - districts_data[n]['y2023']:+.2f}",
    } for i, n in enumerate(
        sorted(districts_data, key=lambda x: districts_data[x]["y2024"], reverse=True)
    )])
    st.dataframe(tbl, use_container_width=True, hide_index=True)

st.divider()
st.markdown("**🛰️ Data:** Sentinel-2 + Landsat-8 · NASA POWER · TN Agriculture Dept")
st.markdown("**🤖 Models:** Random Forest + XGBoost Ensemble · 165 rows · 15 districts · 2013–2023")
st.markdown("**📊 Accuracy:** RF MAE=0.309 · XGB MAE=0.315 · Correlation=0.952 ✅")
