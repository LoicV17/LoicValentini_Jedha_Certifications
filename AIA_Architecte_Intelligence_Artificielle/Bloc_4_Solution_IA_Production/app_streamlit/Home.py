import streamlit as st

st.set_page_config(
    page_title="Lakers Reddit Sentiment – Platform",
    layout="wide",
    page_icon="🏀"
)

# ------------------------------------------------------
# Header principal
# ------------------------------------------------------
st.title("🏀 Lakers Reddit – Plateforme d'Analyse & Monitoring")
st.write(
    "Bienvenue sur la console unifiée du pipeline Reddit → Sentiment → Monitoring. "
    "Utilise le menu latéral pour naviguer entre les modules."
)

# ------------------------------------------------------
# Vue d’ensemble des modules
# ------------------------------------------------------
st.markdown("## 🧭 Navigation générale")

st.markdown(
    """
### 🔍 **Analyse, Monitoring & Industrialisation**

- 📈 **Data Analysis** — courbes temporelles, intensité émotionnelle, LNMI, heatmap et radar  
- 📑 **Data Quality – Evidently** — rapport complet de qualité, distribution et drift  
- 🔧 **CI/CD – Jenkins** — builds, tests, statuts, logs et indicateurs qualité  

---

Chaque module fonctionne indépendamment et contribue à une vision complète du pipeline :
ingestion Reddit → scoring d’émotions → monitoring qualité → CI/CD → orchestration Airflow.
    """
)

# ------------------------------------------------------
# Footer light
# ------------------------------------------------------
st.markdown("---")
st.caption("⚡ Plateforme MLOps – Reddit Lakers • Streamlit • Airflow • Jenkins • Evidently")
