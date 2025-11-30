import streamlit as st
import pandas as pd
import boto3
import os
import altair as alt
import seaborn as sns
import matplotlib.pyplot as plt
import io
import pathlib
from dotenv import load_dotenv
from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError

# Charger les variables du fichier .env (utile en local)
load_dotenv()

# ==============
# STYLE GLOBAL
# ==============
st.set_page_config(page_title="Fraud Detection Report", page_icon="🕵️", layout="wide")

st.markdown("""
    <style>
    /* Police moderne */
    html, body, [class*="css"]  {
        font-family: 'Segoe UI', sans-serif;
    }
    /* KPIs cards */
    .metric-card {
        padding: 15px;
        border-radius: 12px;
        background-color: #f5f7fa;
        text-align: center;
        box-shadow: 0px 2px 5px rgba(0,0,0,0.1);
    }
    .metric-title {
        font-size: 14px;
        color: #666;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #111;
    }
    /* Sidebar branding */
    .sidebar-text {
        color: #aaa;
        font-size: 12px;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# ==============
# CONFIG AWS / SNAPSHOT
# ==============
s3 = boto3.client("s3")
BUCKET = os.getenv("AIRFLOW_S3_BUCKET", "fraud-detection-loicvalentini")
KEY = "reports/full/scored_payments.parquet"

# Snapshot local pour fallback (à ajouter dans ton repo)
SNAPSHOT_PATH = pathlib.Path("data/scored_payments.parquet")


@st.cache_data(show_spinner="Chargement des données de transactions...")
def load_data():
    """
    1️⃣ Essaie de charger les données depuis S3 (source 'prod').
    2️⃣ En cas d'échec (quota, creds, réseau…), bascule sur un snapshot local.
    3️⃣ Ne conserve que les transactions après l'année 2000.
    """
    # --- Lecture des données ---
    try:
        obj = s3.get_object(Bucket=BUCKET, Key=KEY)
        df = pd.read_parquet(io.BytesIO(obj["Body"].read()))

    except (ClientError, BotoCoreError, NoCredentialsError, OSError) as e:
        st.warning("⚠️ Impossible de récupérer les données sur S3. Utilisation d’un snapshot local.")
        st.caption(f"Détail de l’erreur S3 : {e}")

        if not SNAPSHOT_PATH.exists():
            st.error(
                f"❌ Aucun snapshot local trouvé à l’emplacement : {SNAPSHOT_PATH}.\n"
                "Ajoute par exemple un fichier 'data/scored_payments.parquet' ou adapte le chemin."
            )
            raise

        # Lecture du snapshot local (parquet ou csv)
        if SNAPSHOT_PATH.suffix == ".parquet":
            df = pd.read_parquet(SNAPSHOT_PATH)
        else:
            df = pd.read_csv(SNAPSHOT_PATH)

    # --- Filtre : ne garder que les années > 2000 ---
    if "trans_year" in df.columns:
        df = df[df["trans_year"] > 2000].copy()  # change en >= 2000 si besoin

    return df


# ==============
# HEADER + DATA
# ==============
st.title("🕵️ Rapport Fraude Global")
st.markdown("Un aperçu complet des transactions scorées avec détection de fraude.")

# Chargement initial
df = load_data()

if st.button("🔄 Recharger les données"):
    st.cache_data.clear()
    df = load_data()

# ==============
# KPIs
# ==============
fraud_amount = df.loc[df["prediction"] == 1, "amt"].sum()

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.markdown(
        f"<div class='metric-card'><div class='metric-title'>Transactions cumulées</div>"
        f"<div class='metric-value'>{len(df):,}</div></div>",
        unsafe_allow_html=True
    )
with col2:
    st.markdown(
        f"<div class='metric-card'><div class='metric-title'>Fraudes cumulées</div>"
        f"<div class='metric-value'>{df['prediction'].sum():,}</div></div>",
        unsafe_allow_html=True
    )
with col3:
    st.markdown(
        f"<div class='metric-card'><div class='metric-title'>Taux de fraude global</div>"
        f"<div class='metric-value'>{100*df['prediction'].mean():.2f}%</div></div>",
        unsafe_allow_html=True
    )
with col4:
    st.markdown(
        f"<div class='metric-card'><div class='metric-title'>Montant total analysé (€)</div>"
        f"<div class='metric-value'>{df['amt'].sum():,.0f}</div></div>",
        unsafe_allow_html=True
    )
with col5:
    st.markdown(
        f"<div class='metric-card'><div class='metric-title'>Montant total fraudé (€)</div>"
        f"<div class='metric-value' style='color:red;'>{fraud_amount:,.0f}</div></div>",
        unsafe_allow_html=True
    )

st.divider()

# ==========
# VISUELS TEMPORELS
# ==========
st.subheader("📅 Évolution du taux de fraude")

granularity = st.radio("Granularité :", ["Heure", "Jour", "Semaine", "Mois"], horizontal=True)

# Créer une colonne datetime complète (année-mois-jour-heure-minute)
if {"trans_year", "trans_month", "trans_day", "trans_hour"}.issubset(df.columns):
    df["event_time"] = pd.to_datetime(
        df["trans_year"].astype(str) + "-" +
        df["trans_month"].astype(str).str.zfill(2) + "-" +
        df["trans_day"].astype(str).str.zfill(2) + " " +
        df["trans_hour"].astype(str).str.zfill(2) + ":" +
        df.get("trans_minute", 0).astype(str).str.zfill(2),
        errors="coerce"
    )
else:
    st.warning("⚠️ Colonnes temporelles manquantes. Vérifie ton dataset.")
    df["event_time"] = pd.NaT

# Granularité
if granularity == "Heure":
    df["period"] = df["event_time"].dt.to_period("H").apply(lambda r: r.start_time)
elif granularity == "Jour":
    df["period"] = df["event_time"].dt.date
elif granularity == "Semaine":
    df["period"] = df["event_time"].dt.to_period("W").apply(lambda r: r.start_time)
elif granularity == "Mois":
    df["period"] = df["event_time"].dt.to_period("M").apply(lambda r: r.start_time)

fraude_by_period = (
    df.groupby("period")["prediction"].mean().reset_index().dropna()
)

# Convertir en %
fraude_by_period["fraud_rate"] = fraude_by_period["prediction"] * 100

chart = alt.Chart(fraude_by_period).mark_line(point=True).encode(
    x="period:T",
    y=alt.Y("fraud_rate:Q", title="Taux fraude (%)"),
    tooltip=["period", alt.Tooltip("fraud_rate:Q", format=".2f")]
).properties(width=700, height=400)

st.altair_chart(chart, use_container_width=True)

# ==========
# VISUELS ANALYTIQUES
# ==========
st.subheader("🔎 Analyse des fraudes")

col1, col2 = st.columns(2)

# Pie chart par catégorie
with col1:
    fraude_cat = df[df["prediction"] == 1]["category"].value_counts().reset_index()
    fraude_cat.columns = ["category", "count"]

    fig1, ax1 = plt.subplots()
    ax1.pie(fraude_cat["count"], labels=fraude_cat["category"], autopct="%1.1f%%")
    ax1.set_title("Répartition des fraudes par catégorie")
    st.pyplot(fig1)

# Bar chart par état
with col2:
    US_STATES = {
        "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas",
        "CA": "California", "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware",
        "FL": "Florida", "GA": "Georgia", "HI": "Hawaii", "ID": "Idaho",
        "IL": "Illinois", "IN": "Indiana", "IA": "Iowa", "KS": "Kansas",
        "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland",
        "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi",
        "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada",
        "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York",
        "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma",
        "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina",
        "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas", "UT": "Utah",
        "VT": "Vermont", "VA": "Virginia", "WA": "Washington", "WV": "West Virginia",
        "WI": "Wisconsin", "WY": "Wyoming"
    }

    df["state_full"] = df["state"].map(US_STATES).fillna(df["state"])

    fraudes_par_state = df[df["prediction"] == 1]["state_full"].value_counts().reset_index()
    fraudes_par_state.columns = ["État", "Nombre de fraudes"]

    chart_state = alt.Chart(fraudes_par_state).mark_bar(color="red").encode(
        x="Nombre de fraudes:Q",
        y=alt.Y("État:N", sort="-x"),
        tooltip=["État", "Nombre de fraudes"]
    ).properties(width=350, height=350, title="Fraudes par État")

    st.altair_chart(chart_state, use_container_width=True)

# ==========
# DATASET COMPLET
# ==========
df_display = df.rename(columns={"unnamed_0": "trans_number"})
df_display["date"] = df_display["event_time"].dt.strftime("%Y-%m-%d %H:%M")

cols_order = ["trans_number", "date", "amt", "probability", "state_full"]
other_cols = [c for c in df_display.columns if c not in cols_order]
df_display = df_display[cols_order + other_cols]

st.subheader("📂 Détails des fraudes détectées")
fraude_details = df_display[df_display["prediction"] == 1]
st.dataframe(fraude_details)

# Bouton téléchargement
csv = fraude_details.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Télécharger CSV complet", data=csv, file_name="fraudes.csv", mime="text/csv")

# ==========
# SIDEBAR
# ==========
with st.sidebar:
    st.markdown("---")
    st.markdown(
        "<span class='sidebar-text'>"
        "Made by <b>Loic Valentini</b><br>"
        "Jedha AIA – Projet Bloc 3 – <i>Fraud_detection</i>"
        "</span>",
        unsafe_allow_html=True
    )
