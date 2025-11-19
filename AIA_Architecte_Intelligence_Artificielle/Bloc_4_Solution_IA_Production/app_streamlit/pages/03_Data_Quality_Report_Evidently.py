import streamlit as st
from pathlib import Path
from bs4 import BeautifulSoup
import os

st.title("📊 Evidently – Dernier rapport de qualité")

# -------------------------------------------------------------------
# 1) Localisation automatique du dossier des rapports
# -------------------------------------------------------------------

# /app_streamlit/pages/02_...py  → remonte de 2 niveaux vers ../
BASE_DIR = Path(__file__).resolve().parents[2]

REPORT_DIR = BASE_DIR / "data" / "reports" / "evidently"

st.write(f"🔎 Chemin utilisé : `{REPORT_DIR}`")

if not REPORT_DIR.exists():
    st.error("❌ Dossier inexistant.")
    st.stop()

# -------------------------------------------------------------------
# 2) Récupération du dernier rapport HTML Evidently
# -------------------------------------------------------------------

html_files = sorted([f for f in REPORT_DIR.glob("*.html")], reverse=True)

if not html_files:
    st.error("❌ Aucun rapport Evidently trouvé dans data/reports/evidently.")
    st.stop()

latest_html = html_files[0]
st.success(f"📄 Rapport détecté : {latest_html.name}")

# -------------------------------------------------------------------
# 3) Lecture du contenu HTML
# -------------------------------------------------------------------

with open(latest_html, "r", encoding="utf-8") as f:
    html_content = f.read()

# Optionnel : nettoyage du HTML
soup = BeautifulSoup(html_content, "html.parser")
clean_html = str(soup)

# -------------------------------------------------------------------
# 4) Affichage dans Streamlit
# -------------------------------------------------------------------

st.components.v1.html(clean_html, height=1200, scrolling=True)
