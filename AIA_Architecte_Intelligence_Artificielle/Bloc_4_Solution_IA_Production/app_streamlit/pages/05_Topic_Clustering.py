# app_streamlit/pages/05_Topic_Clustering.py

import os
import pandas as pd
import streamlit as st
from sqlalchemy import create_engine
from dotenv import load_dotenv

# -------------------------
# Configuration
# -------------------------
load_dotenv()
DB_URL = os.getenv("DATABASE_URL")

st.set_page_config(page_title="Topic Clustering Reddit", layout="wide")
st.title("🧩 Clustering thématique des posts Reddit")

if DB_URL is None:
    st.error("DATABASE_URL non défini.")
    st.stop()

engine = create_engine(DB_URL)


# -------------------------
# Sidebar : filtres
# -------------------------
st.sidebar.header("Filtres")

period_label = st.sidebar.selectbox(
    "Période",
    ["7 derniers jours", "30 derniers jours", "Tous les messages"],
    index=0
)

if period_label == "7 derniers jours":
    period_days = 7
elif period_label == "30 derniers jours":
    period_days = 30
else:
    period_days = None   # Tous les messages


# -------------------------
# Chargement des données
# -------------------------
@st.cache_data(ttl=300)
def load_posts(days):
    base_query = """
        SELECT
            id,
            created_utc,
            title,
            selftext,
            emotion_main,
            anger,
            disgust,
            fear,
            joy,
            neutral,
            sadness,
            surprise,
            topic_cluster
        FROM reddit_enriched
    """

    if days is None:
        query = base_query + " ORDER BY created_utc DESC"
    else:
        query = base_query + f"""
            WHERE created_utc >= NOW() - INTERVAL '{days} days'
            ORDER BY created_utc DESC
        """
    return pd.read_sql(query, engine)


df = load_posts(period_days)

if df.empty:
    st.info("Aucun post disponible pour la période sélectionnée.")
    st.stop()

df["text"] = df["title"].fillna("") + " " + df["selftext"].fillna("")


# -------------------------
# Calcul LNMI
# -------------------------
df["LNMI"] = (
    df["joy"] + df["surprise"]
    - (df["anger"] + df["fear"] + df["sadness"] + df["disgust"])
)

lnmi_cluster = df.groupby("topic_cluster")["LNMI"].mean().to_frame()


# -------------------------
# Titres humains des clusters
# -------------------------
cluster_titles = {
    0: "Hors-sujet / Bruit",
    1: "Analyse de jeu & joueurs",
    2: "LeBron & réactions aux matchs",
    3: "Luka Doncic — Performance / MVP talk",
    4: "Fan talk émotionnel / WTF",
    5: "Matchday vibe / Bronny / hype",
    6: "Threads quotidiens / méta-discussion",
    7: "Trade talk / Legacy / Lakers"
}

cluster_keywords = {
    0: "texas, feeding, ad",
    1: "game, ayton, luka, vando, rui",
    2: "lebron, james, nba, reaves",
    3: "luka, doncic, mvp, shooting",
    4: "bron, reaves, love, wtf",
    5: "tonight, vibes, bronny",
    6: "daily thread, discussion, meta",
    7: "lakers, trade, kobe, team"
}


# -------------------------
# Section 1 : Clusters identifiés
# -------------------------
st.subheader("📌 Clusters identifiés")
st.info("Vue d’ensemble des thèmes détectés automatiquement, avec titre humain + mots-clés.")

cluster_df = pd.DataFrame([
    {
        "Cluster": k,
        "Titre humain": cluster_titles[k],
        "Mots-clés": cluster_keywords[k]
    }
    for k in sorted(cluster_titles.keys())
])

st.dataframe(cluster_df, use_container_width=True)

st.markdown("---")


# -------------------------
# Section 2 : Nb de posts par cluster
# -------------------------
st.subheader("📊 Nombre de posts par cluster")
st.info("Montre les thématiques les plus discutées.")

cluster_counts = df["topic_cluster"].value_counts().sort_index()
st.bar_chart(cluster_counts)

st.markdown("---")


# -------------------------
# Section 3 : LNMI global par cluster
# -------------------------
st.subheader("💛 LNMI — Lakers Nation Mood Index par cluster")
st.info("Plus le LNMI moyen est élevé, plus le ton global de la communauté est positif dans ce thème.")

lnmi_cluster["Titre humain"] = lnmi_cluster.index.map(cluster_titles)
lnmi_cluster = lnmi_cluster.rename(columns={"LNMI": "LNMI_moyen"})
lnmi_cluster = lnmi_cluster[["Titre humain", "LNMI_moyen"]]

st.dataframe(lnmi_cluster, use_container_width=True)
st.bar_chart(lnmi_cluster["LNMI_moyen"])

st.markdown("---")

