from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from pathlib import Path
import subprocess
import logging
import sys

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
    description="Pipeline Reddit Lakers - Ingestion + Nettoyage + Scoring + Data Quality",
    schedule_interval=timedelta(hours=1),
    start_date=datetime(2025, 10, 30),
    catchup=False,
    max_active_runs=1,
    tags=["reddit", "nba", "lakers"],
) as dag:

    PROJECT_ROOT = Path(__file__).resolve().parents[1]

    def run_script(script_relative_path, description):
        """Exécute un script Python en sous-processus avec logs propres."""
        logging.info("────────────────────────────────────────────")
        logging.info(f"🚀 Lancement : {description}")
        logging.info("────────────────────────────────────────────")

        script_path = PROJECT_ROOT / "src" / script_relative_path

        # Vérification script existant
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


    # --- Étape 1 : Ingestion Reddit ---
    task_fetch = PythonOperator(
        task_id="fetch_reddit_posts",
        python_callable=lambda: run_script(
            "ingestion/fetch_reddit_lakers.py",
            "Ingestion Reddit"
        )
    )

    # --- Étape 2 : Nettoyage ---
    task_clean = PythonOperator(
        task_id="clean_reddit_data",
        python_callable=lambda: run_script(
            "preprocessing/clean_reddit_data.py",
            "Nettoyage des données Reddit"
        )
    )

    # --- Étape 3 : Chargement NeonDB ---
    task_load = PythonOperator(
        task_id="load_to_neondb",
        python_callable=lambda: run_script(
            "load/load_to_neondb.py",
            "Chargement vers NeonDB"
        )
    )

    # --- Étape 4 : Scoring émotions ---
    task_score_emotions = PythonOperator(
        task_id="score_emotions",
        python_callable=lambda: run_script(
            "ml/score_emotions.py",
            "Scoring émotionnel des posts Reddit"
        )
    )

    # --- Étape 5 : Rapport Evidently ---
    task_data_quality_report = PythonOperator(
        task_id="task_data_quality_report",
        python_callable=lambda: run_script(
            "ml/generate_data_quality_report.py",
            "Rapport Evidently Data Quality"
        )
    )

    # --- Étape 6 : Insertion historique ---
    task_insert_quality_history = PythonOperator(
        task_id="task_insert_quality_history",
        python_callable=lambda: run_script(
            "ml/insert_quality_metrics.py",
            "Insertion des métriques Illumination"
        )
    )

    # --- Orchestration du pipeline ---
    task_fetch >> task_clean >> task_load >> task_score_emotions >> task_data_quality_report >> task_insert_quality_history
