import streamlit as st
import requests
import pandas as pd
from datetime import datetime
import base64

st.set_page_config(page_title="Airflow Monitoring", layout="wide")
st.title("🛠 Airflow – Monitoring & Debug Console")

# -------------------------------------------------------------
# CONFIG – Airflow API parameters
# -------------------------------------------------------------
AIRFLOW_URL = "http://localhost:8085"
AIRFLOW_API_BASE = f"{AIRFLOW_URL}/api/v1"

AIRFLOW_USER = "admin"
AIRFLOW_PASSWORD = "admin"

# Basic Auth Header
auth_header = base64.b64encode(f"{AIRFLOW_USER}:{AIRFLOW_PASSWORD}".encode()).decode()
HEADERS = {"Authorization": f"Basic {auth_header}"}


# -------------------------------------------------------------
# Charger les DAGs
# -------------------------------------------------------------
try:
    response = requests.get(f"{AIRFLOW_API_BASE}/dags", headers=HEADERS, timeout=5)
    response.raise_for_status()
except Exception as e:
    st.error("❌ Impossible de contacter Airflow (port 8085 ? Docker lancé ?)")
    st.text(str(e))
    st.stop()

dag_ids = [d["dag_id"] for d in response.json().get("dags", [])]
selected_dag = st.selectbox("📋 Sélection du DAG", dag_ids)


# -------------------------------------------------------------
# Charger les RUNS (max 100)
# -------------------------------------------------------------
runs_url = (
    f"{AIRFLOW_API_BASE}/dags/{selected_dag}/dagRuns"
    "?limit=100&order_by=-execution_date"
)

runs_resp = requests.get(runs_url, headers=HEADERS)
dag_runs = runs_resp.json().get("dag_runs", [])


# Utilitaires formatage dates
def fmt_dt(s):
    if not s:
        return ""
    return datetime.fromisoformat(s.replace("Z", "+00:00")).strftime("%Y-%m-%d %H:%M")


# Ajout des dates formatées
for r in dag_runs:
    r["execution_date_fmt"] = fmt_dt(r.get("execution_date"))
    r["start_date_fmt"] = fmt_dt(r.get("start_date"))
    r["end_date_fmt"] = fmt_dt(r.get("end_date"))
    r["run_id_final"] = r.get("run_id", r.get("dag_run_id"))


# -------------------------------------------------------------
# COULEURS STATUT
# -------------------------------------------------------------
status_icon = {
    "success": "🟩",
    "failed": "🟥",
    "running": "🟦",
    "queued": "🟧",
    "up_for_retry": "🟨",
}

# -------------------------------------------------------------
# TIMELINE (10 derniers)
# -------------------------------------------------------------
st.subheader("📊 Timeline – 10 derniers runs")

timeline_html = ""
for r in dag_runs[:10]:  # déjà triés descendants
    icon = status_icon.get(r["state"], "⬜")
    timeline_html += f"{icon} <b>{r['execution_date_fmt']}</b><br><br>"

st.markdown(timeline_html, unsafe_allow_html=True)


# -------------------------------------------------------------
# TABLEAU DÉTAILLÉ (100 derniers runs, + récent → ancien)
# -------------------------------------------------------------
st.subheader("📋 Tableau détaillé (100 derniers runs)")

df_display = pd.DataFrame([
    {
        "Status": status_icon.get(r["state"], "⬜"),
        "Execution Date": r["execution_date_fmt"],
        "Start": r["start_date_fmt"],
        "End": r["end_date_fmt"],
        "Run ID": r["run_id_final"],
    }
    for r in dag_runs
])

st.dataframe(df_display, use_container_width=True)



# -------------------------------------------------------------
# 🔥 LOGS COMPLETS DU RUN – toutes les tasks automatiquement
# -------------------------------------------------------------
st.subheader("📜 Logs complets du Run sélectionné")

# Choix du run
run_choices = {r["execution_date_fmt"]: r["run_id_final"] for r in dag_runs}

selected_run_label = st.selectbox("📌 Choisir un run", list(run_choices.keys()))
selected_run_id = run_choices[selected_run_label]

if selected_run_id:

    tasks_url = f"{AIRFLOW_API_BASE}/dags/{selected_dag}/dagRuns/{selected_run_id}/taskInstances"
    tasks_resp = requests.get(tasks_url, headers=HEADERS)

    if tasks_resp.status_code != 200:
        st.error("❌ Impossible de récupérer les tasks pour ce run.")
        st.text(tasks_resp.text)
    else:
        task_list = tasks_resp.json().get("task_instances", [])

        if not task_list:
            st.info("Aucune tâche trouvée pour ce Run.")
        else:
            st.markdown(f"### 🔧 Run `{selected_run_id}` – {len(task_list)} tâches analysées")

            # Parcours toutes les tâches pour récupérer les logs
            for task in task_list:
                task_id = task["task_id"]

                st.markdown(f"#### ▶ Task : **{task_id}**")

                log_url = (
                    f"{AIRFLOW_API_BASE}/dags/{selected_dag}/dagRuns/{selected_run_id}"
                    f"/taskInstances/{task_id}/logs/1?full_content=true"
                )

                log_resp = requests.get(log_url, headers=HEADERS)

                if log_resp.status_code == 200:
                    try:
                        log_data = log_resp.json()
                        st.code(log_data.get("content", ""), language="bash")
                    except:
                        st.code(log_resp.text, language="bash")
                else:
                    st.error(f"❌ Impossible de charger les logs de la tâche {task_id}")
                    st.text(log_resp.text)

