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

    def run_fetch_and_push_latest(**context):
        """Exécute le fetch puis pousse le dernier CSV brut via XCom."""
        script_path = PROJECT_ROOT / "src" / "ingestion" / "fetch_reddit_lakers.py"
        logging.info("🚀 Fetch Reddit...")
        res = subprocess.run(
            ["python", str(script_path)],
            capture_output=True, text=True
        )
        if res.returncode != 0:
            logging.error("STDOUT:\n" + (res.stdout or ""))
            logging.error("STDERR:\n" + (res.stderr or ""))
            raise RuntimeError("Échec du script d'ingestion Reddit.")
        logging.info(res.stdout)

        # Choisir le fichier le plus récent dans data/raw
        raw_files = sorted(RAW_DIR.glob("reddit_*.csv"), key=lambda f: f.stat().st_mtime, reverse=True)
        if not raw_files:
            raise RuntimeError("Aucun fichier brut trouvé après fetch.")
        latest = str(raw_files[0])
        logging.info(f"🆕 Dernier fichier brut détecté: {latest}")
        context["ti"].xcom_push(key="latest_file", value=latest)
        return latest

    def run_clean_with_xcom(**context):
        """Récupère le chemin du dernier CSV brut et lance le nettoyage incrémental."""
        latest_file = context["ti"].xcom_pull(task_ids="fetch_reddit_posts", key="latest_file")
        if not latest_file:
            raise RuntimeError("Chemin du fichier brut introuvable en XCom.")

        script_path = PROJECT_ROOT / "src" / "preprocessing" / "clean_reddit_data.py"
        logging.info(f"🧼 Nettoyage du fichier {latest_file}")
        res = subprocess.run(
            ["python", str(script_path), "--latest-file", latest_file],
            capture_output=True, text=True
        )
        if res.returncode != 0:
            logging.error("STDOUT:\n" + (res.stdout or ""))
            logging.error("STDERR:\n" + (res.stderr or ""))
            raise RuntimeError("Échec du nettoyage Reddit.")
        logging.info(res.stdout)
        logging.info("✅ Nettoyage terminé avec succès.")

    task_fetch_reddit = PythonOperator(
        task_id="fetch_reddit_posts",
        python_callable=run_fetch_and_push_latest,
        provide_context=True,
    )

    task_clean_reddit = PythonOperator(
        task_id="clean_reddit_data",
        python_callable=run_clean_with_xcom,
        provide_context=True,
    )

    task_fetch_reddit >> task_clean_reddit

dag
