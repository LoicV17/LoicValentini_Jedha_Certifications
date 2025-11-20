# dags/reddit_topic_retrain_dag.py

from datetime import datetime, timedelta
from pathlib import Path
import logging
import subprocess
import sys

from airflow import DAG
from airflow.operators.python import PythonOperator

default_args = {
    "owner": "loic_valentini",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
}

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_script(script_relative_path, description):
    """Exécute un script Python en sous-processus avec logs propres."""
    logging.info("────────────────────────────────────────────")
    logging.info(f"🚀 Lancement : {description}")
    logging.info("────────────────────────────────────────────")

    script_path = PROJECT_ROOT / "src" / script_relative_path

    if not script_path.exists():
        raise FileNotFoundError(f"❌ Script introuvable : {script_path}")

    result = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        logging.error("❌ ERREUR pendant l’exécution")
        logging.error(result.stderr)
        raise RuntimeError(f"❌ Erreur dans : {description}")

    logging.info(result.stdout)
    logging.info(f"✅ Terminé : {description}")
    logging.info("────────────────────────────────────────────")
    return True


with DAG(
    dag_id="reddit_topic_retrain_dag",
    default_args=default_args,
    description="Réentraînement quotidien du modèle de clustering thématique Reddit",
    schedule_interval="0 */4 * * *",   # ⬅️ toutes les 4 heures
    start_date=datetime(2025, 11, 20),
    catchup=False,
    max_active_runs=1,
    tags=["reddit", "nba", "lakers", "topics", "retrain"],
) as dag:


    retrain_topics = PythonOperator(
        task_id="retrain_topic_model",
        python_callable=lambda: run_script(
            "ml/train_topic_model_daily.py",
            "Réentraînement du modèle de topic clustering"
        ),
    )

    retrain_topics
