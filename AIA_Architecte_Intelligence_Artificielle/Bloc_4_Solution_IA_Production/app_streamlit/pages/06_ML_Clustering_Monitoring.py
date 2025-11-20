import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
from dotenv import load_dotenv
load_dotenv()


# -----------------------------------------------------------
# DB : load metrics + posts
# -----------------------------------------------------------
DB_URL = os.getenv("DATABASE_URL")
engine = create_engine(DB_URL)

@st.cache_data
def load_history():
    query = """
        SELECT *
        FROM reddit_topic_models_metrics
        ORDER BY created_at DESC
    """
    return pd.read_sql(query, engine)

@st.cache_data
def load_posts():
    query = """
        SELECT id, title, selftext,
               COALESCE(title,'') || ' ' || COALESCE(selftext,'') AS text
        FROM reddit_cleaned
    """
    return pd.read_sql(query, engine)

# -----------------------------------------------------------
# Page
# -----------------------------------------------------------
st.title("🔁 Réentraînement & Monitoring — Topic Clustering")

st.info("""
Cette page regroupe **l’historique complet**, la sélection du modèle,
les **métriques**, les **dérives**, et l’exploration des **clusters**
(avec mots-clés + 3 posts représentatifs).
""")

# ─────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────
df_hist = load_history()
df_posts = load_posts()

if df_hist.empty:
    st.error("Aucun modèle trouvé dans l’historique.")
    st.stop()

# ─────────────────────────────────────────────────────────────
# CHOIX DU MODÈLE
# ─────────────────────────────────────────────────────────────
st.subheader("📌 Sélection du modèle")

model_idx = st.selectbox(
    "Choisis un modèle",
    options=df_hist.index,
    format_func=lambda i: f"{df_hist.loc[i,'created_at']} — {df_hist.loc[i,'model_path']}",
)

selected = df_hist.loc[model_idx]

model_path = selected["model_path"]
n_clusters = selected["n_clusters"]
cluster_sizes = selected["cluster_sizes"]
cluster_keywords = selected["cluster_keywords"]


# ─────────────────────────────────────────────────────────────
# METRIQUES GLOBALES
# ─────────────────────────────────────────────────────────────
st.subheader("📊 Métriques du modèle sélectionné")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Silhouette", f"{selected['silhouette']:.4f}")
col2.metric("Inertia", f"{selected['inertia']:.0f}")
col3.metric("Clusters", int(selected["n_clusters"]))
col4.metric("Posts", int(selected["n_samples"]))

# ─────────────────────────────────────────────────────────────
# DERIVE DANS LE TEMPS
# ─────────────────────────────────────────────────────────────
st.subheader("📈 Évolution temporelle des métriques")

def plot_metric(df, column):
    fig, ax = plt.subplots(figsize=(5,2))
    ax.plot(df["created_at"], df[column], marker="o")
    ax.set_title(column)
    ax.grid(alpha=0.3)
    return fig

colA, colB, colC = st.columns(3)
with colA:
    st.pyplot(plot_metric(df_hist, "silhouette"))
with colB:
    st.pyplot(plot_metric(df_hist, "inertia"))
with colC:
    st.pyplot(plot_metric(df_hist, "n_clusters"))


st.success("✔ Page chargée avec succès")
