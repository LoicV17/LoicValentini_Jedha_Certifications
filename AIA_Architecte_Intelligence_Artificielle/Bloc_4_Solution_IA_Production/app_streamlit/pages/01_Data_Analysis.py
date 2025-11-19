import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np
from sqlalchemy import create_engine
from datetime import timedelta
import os
from dotenv import load_dotenv

load_dotenv()

# ------------------------------------------------------------
# Connexion DB
# ------------------------------------------------------------
DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)

if DATABASE_URL is None:
    st.error("❌ DATABASE_URL introuvable. Vérifie ton fichier .env.")
    st.stop()

@st.cache_data(ttl=300)
def load_data():
    query = """
        SELECT id, created_at, joy, anger, sadness, surprise, disgust, fear, neutral, main_emotion
        FROM reddit_scoring
        ORDER BY created_at ASC;
    """
    return pd.read_sql(query, engine)

df = load_data()
df["created_at"] = pd.to_datetime(df["created_at"])

emotion_cols = ["joy","anger","sadness","surprise","disgust","fear","neutral"]

st.title("📊 Data Analysis")
st.markdown("---")


# =====================================================================
# 1️⃣ KPI GLOBAUX (NON FILTRÉS)
# =====================================================================
st.header("🌍 Indicateurs globaux")

df["sentiment"] = df["joy"] - df["anger"]

now = df["created_at"].max()
last_hour = now - timedelta(hours=1)
last_24h = now - timedelta(hours=24)

df_1h = df[df["created_at"] >= last_hour].copy()
df_24h = df[df["created_at"] >= last_24h].copy()

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total posts", len(df))

with col2:
    st.metric("Dernière heure", len(df_1h))

with col3:
    st.metric("24 dernières heures", len(df_24h))

# Émotions dominantes
col4, col5 = st.columns(2)
emotion_1h = df_1h["main_emotion"].mode()[0] if len(df_1h) else "N/A"
emotion_24h = df_24h["main_emotion"].mode()[0] if len(df_24h) else "N/A"

with col4:
    st.metric("Émotion dominante (1h)", emotion_1h)

with col5:
    st.metric("Émotion dominante (24h)", emotion_24h)

st.markdown("---")


# =====================================================================
# 2️⃣ FILTRE RAPIDE (SIDEBAR) + SLIDER
# =====================================================================
st.sidebar.header("⏱️ Filtre rapide")

quick = st.sidebar.radio(
    "Choisir une période",
    ["Tout", "1h", "6h", "12h", "24h", "7 jours"],
)

now = df["created_at"].max()

if quick == "1h":
    start_quick = now - timedelta(hours=1)
elif quick == "6h":
    start_quick = now - timedelta(hours=6)
elif quick == "12h":
    start_quick = now - timedelta(hours=12)
elif quick == "24h":
    start_quick = now - timedelta(hours=24)
elif quick == "7 jours":
    start_quick = now - timedelta(days=7)
else:
    start_quick = df["created_at"].min()

st.subheader("⏳ Période d'analyse (filtre manuel)")

# ------------------------------------------------------------
# Harmonisation des types : Streamlit n'accepte PAS les Timestamps
# ------------------------------------------------------------
if isinstance(start_quick, pd.Timestamp):
    start_quick = start_quick.to_pydatetime()

min_date = df["created_at"].min()
max_date = df["created_at"].max()

if isinstance(min_date, pd.Timestamp):
    min_date = min_date.to_pydatetime()

if isinstance(max_date, pd.Timestamp):
    max_date = max_date.to_pydatetime()

# ------------------------------------------------------------
# Slider
# ------------------------------------------------------------
start_date, end_date = st.slider(
    "Sélectionne la période",
    min_value=min_date,
    max_value=max_date,
    value=(start_quick, max_date),
    step=timedelta(hours=1),
)


df_filtered = df[(df["created_at"] >= start_date) & (df["created_at"] <= end_date)]

st.markdown("---")


# =====================================================================
# 3️⃣ INDICATEURS BASÉS SUR PÉRIODE FILTRÉE
# =====================================================================

# ----------------------------
# LNMI
# ----------------------------
st.header("📈 Indicateurs temporels")

st.subheader("💛 LNMI — Lakers Nation Mood Index")

lnmi = (
    df_filtered["joy"].mean() +
    df_filtered["surprise"].mean()
    - df_filtered["anger"].mean()
    - df_filtered["fear"].mean()
    - df_filtered["sadness"].mean()
    - df_filtered["disgust"].mean()
)

st.metric("LNMI", f"{lnmi:.3f}")

st.info("""
LNMI = (joie + surprise) – (colère + peur + tristesse + dégoût).  
> 0 → humeur positive  
≈ 0 → neutre  
< 0 → négative  
""")

st.markdown("---")


# ----------------------------
# Courbe LNMI
# ----------------------------
st.subheader("📈 LNMI – évolution horaire")

df_lnmi = df_filtered.copy()
df_lnmi["hour"] = df_lnmi["created_at"].dt.floor("H")
df_lnmi["lnmi"] = (
    df_lnmi["joy"] + df_lnmi["surprise"]
    - df_lnmi["anger"] - df_lnmi["fear"]
    - df_lnmi["sadness"] - df_lnmi["disgust"]
)

df_lnmi = df_lnmi.groupby("hour")["lnmi"].mean().reset_index()

fig_lnmi = px.line(df_lnmi, x="hour", y="lnmi", title="LNMI")
fig_lnmi.update_layout(template="simple_white")

st.plotly_chart(fig_lnmi, use_container_width=True)

st.markdown("---")


# ----------------------------
# Courbe multi-émotions
# ----------------------------
st.subheader("📈 Émotions par heure")

df_hourly = df_filtered.copy()
df_hourly["hour"] = df_hourly["created_at"].dt.floor("H")
df_hourly = df_hourly.groupby("hour")[emotion_cols].mean().reset_index()

fig = px.line(df_hourly, x="hour", y=emotion_cols, title="Émotions")
fig.update_layout(template="simple_white")

st.plotly_chart(fig, use_container_width=True)

st.markdown("---")


# ----------------------------
# Heatmap
# ----------------------------
st.subheader("🔥 Heatmap du sentiment")

df_heat = df_filtered.copy()
df_heat["hour"] = df_heat["created_at"].dt.hour
df_heat["day"] = df_heat["created_at"].dt.date

heat = df_heat.pivot_table(
    index="hour",
    columns="day",
    values="sentiment",
    aggfunc="mean"
)

fig_heat = px.imshow(
    heat,
    aspect="auto",
    color_continuous_scale="RdBu",
    title="Heatmap"
)
fig_heat.update_layout(template="simple_white")
st.plotly_chart(fig_heat, use_container_width=True)

st.markdown("---")


# ----------------------------
# Radar compact & centré
# ----------------------------
st.subheader("🛑 Radar des émotions moyennes")

values = df_filtered[emotion_cols].mean().values
angles = np.linspace(0, 2*np.pi, len(values), endpoint=False)

values = np.concatenate((values, [values[0]]))
angles = np.concatenate((angles, [angles[0]]))

fig_radar, ax = plt.subplots(subplot_kw={'projection':'polar'}, figsize=(2.5,2.5))
ax.plot(angles, values, linewidth=2)
ax.fill(angles, values, alpha=0.25)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(emotion_cols, fontsize=10)
ax.set_yticklabels([])
ax.grid(True)

col_center = st.columns([1,2,1])[1]

with col_center:
    st.pyplot(fig_radar)
