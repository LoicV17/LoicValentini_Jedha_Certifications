from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from pathlib import Path
import subprocess
import logging

default_args = {
    "owner": "loic_valentini",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="reddit_pipeline_dag",
    default_args=default_args,
    description="Pipeline Reddit Lakers - Ingestion + Nettoyage",
    schedule_interval=timedelta(hours=1),
    start_date=datetime(2025, 10, 30),
    catchup=False,
    max_active_runs=1,
    tags=["reddit", "nba", "lakers"],
) as dag:

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    RAW_DIR = PROJECT_ROOT / "data" / "raw"

    def run_script(script_relative_path, description):
        """Helper pour exécuter un script Python."""
        logging.info(f"🚀 Lancement de {description} ...")
        project_root = Path(__file__).resolve().parents[1]
        script_path = project_root / "src" / script_relative_path

        result = subprocess.run(
            ["python", str(script_path)],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            logging.error(result.stderr)
            raise RuntimeError(f"❌ Erreur dans {description}")

        logging.info(result.stdout)
        logging.info(f"✅ {description} terminé avec succès.")
        return True


    # --- Étape 1 : Ingestion Reddit ---
    task_fetch = PythonOperator(
        task_id='fetch_reddit_posts',
        python_callable=lambda: run_script("ingestion/fetch_reddit_lakers.py", "Ingestion Reddit")
    )

    # --- Étape 2 : Nettoyage ---
    task_clean = PythonOperator(
        task_id='clean_reddit_data',
        python_callable=lambda: run_script("preprocessing/clean_reddit_data.py", "Nettoyage des données Reddit")
    )

    # --- Étape 3 : Chargement NeonDB ---
    task_load = PythonOperator(
        task_id='load_to_neondb',
        python_callable=lambda: run_script("load/load_to_neondb.py", "Chargement vers NeonDB")
    )

    # --- Orchestration ---
    task_fetch >> task_clean >> task_load
