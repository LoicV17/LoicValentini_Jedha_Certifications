import os
import requests
import streamlit as st
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

# ---------------------------------------------------------
# 🔐 Charger variables d’environnement
# ---------------------------------------------------------
load_dotenv()

JENKINS_URL = os.getenv("JENKINS_URL")
JOB_NAME = os.getenv("JENKINS_JOB")
USERNAME = os.getenv("JENKINS_USER")
API_TOKEN = os.getenv("JENKINS_TOKEN")

auth = (USERNAME, API_TOKEN)

st.title("🧩 Jenkins – Suivi CI/CD du Pipeline Reddit")


# ---------------------------------------------------------
# 🔧 Helper pour appeler l’API Jenkins
# ---------------------------------------------------------
def get_json(url):
    r = requests.get(url, auth=auth, timeout=10)
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------
# 🟦 SECTION 1 — KPIs généraux
# ---------------------------------------------------------
st.header("📊 État général du job Jenkins")

try:
    data = get_json(f"{JENKINS_URL}/job/{JOB_NAME}/api/json")

    last_build = data["lastBuild"]["number"]
    last_success = data["lastSuccessfulBuild"]["number"]
    last_failed = data["lastFailedBuild"]["number"]

    col1, col2, col3 = st.columns(3)
    col1.metric("Dernier build", last_build)
    col2.metric("Dernier succès", last_success)
    col3.metric("Dernier échec", last_failed)

except Exception as e:
    st.error("❌ Impossible de récupérer les informations Jenkins.")
    st.exception(e)
    st.stop()


# ---------------------------------------------------------
# 🟧 SECTION 2 — Historique des builds
# ---------------------------------------------------------
st.header("📜 Historique des builds")

try:
    builds = data["builds"][:20]  # derniers 20 builds
    rows = []

    for b in builds:
        info = get_json(f"{JENKINS_URL}/job/{JOB_NAME}/{b['number']}/api/json")
        ts = datetime.fromtimestamp(info["timestamp"] / 1000)
        duration = info["duration"] / 1000

        rows.append({
            "Build": b["number"],
            "Date": ts.strftime("%Y-%m-%d %H:%M"),
            "Durée (s)": round(duration, 1),
            "Résultat": info.get("result", "IN_PROGRESS"),
        })

    df_builds = pd.DataFrame(rows)
    st.dataframe(df_builds)

except Exception as e:
    st.error("❌ Impossible de charger l'historique.")
    st.exception(e)


# ---------------------------------------------------------
# 🟨 SECTION 3 — Tests détaillés (avec session_state FIX)
# ---------------------------------------------------------
st.header("🧪 Détail des tests")

build_number = st.number_input("Numéro du build", min_value=1, step=1, value=last_build)

# Initialisation état
if "tests_loaded" not in st.session_state:
    st.session_state.tests_loaded = False

if "df_tests" not in st.session_state:
    st.session_state.df_tests = None


# 🔘 Bouton pour charger les tests
if st.button("Charger les tests"):
    try:
        report = get_json(
            f"{JENKINS_URL}/job/{JOB_NAME}/{build_number}/testReport/api/json"
        )

        all_tests = []
        for suite in report.get("suites", []):
            for case in suite.get("cases", []):
                status = case["status"]

                if status == "PASSED":
                    icon = "✔️"
                    color = "🟢"
                elif status == "FAILED":
                    icon = "✖️"
                    color = "🔴"
                else:
                    icon = "➖"
                    color = "🟡"

                all_tests.append({
                    "Test": case["name"],
                    "Classe": case["className"],
                    "Statut": f"{color} {icon} {status}",
                    "Durée (s)": round(case["duration"], 3),
                    "Erreur": case.get("errorDetails"),
                })

        st.session_state.df_tests = pd.DataFrame(all_tests)
        st.session_state.tests_loaded = True

    except Exception as e:
        st.error(f"❌ Aucun test trouvé pour le build {build_number}.")
        st.exception(e)


# -------------------------------------------
# 🔄 Affichage stable même après rerun
# -------------------------------------------
if st.session_state.tests_loaded and st.session_state.df_tests is not None:

    df_tests = st.session_state.df_tests

    st.subheader(f"🔍 {len(df_tests)} tests dans le build {build_number}")
    st.dataframe(df_tests, use_container_width=True)

    test_selected = st.selectbox("Voir les logs d'un test :", df_tests["Test"])

    if test_selected:
        err = df_tests[df_tests["Test"] == test_selected]["Erreur"].values[0]
        if err:
            st.error(err)
        else:
            st.success("Aucune erreur 👍")

