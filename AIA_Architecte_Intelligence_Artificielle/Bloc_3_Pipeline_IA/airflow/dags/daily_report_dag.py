from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.hooks.base import BaseHook
from datetime import datetime, timedelta
from sqlalchemy import create_engine
import pandas as pd

# ======================
# 📅 Paramètres du DAG
# ======================
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": True,
    "email": ["valentini.loic@gmail.com"],  # mets ton adresse
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

# ======================
# 📂 Fonctions
# ======================
def generate_report(**kwargs):
    # 🔐 Récupérer connexion NeonDB depuis Airflow UI (Admin > Connections)
    conn = BaseHook.get_connection("neondb")  # nom du Conn Id dans Airflow

    # ✅ Construire l’URL SQLAlchemy avec psycopg2
    DB_URL = f"postgresql+psycopg2://{conn.login}:{conn.password}@{conn.host}:{conn.port}/{conn.schema}"
    engine = create_engine(DB_URL)

    # 📅 Calculer la date de la veille
    yesterday = (datetime.now() - timedelta(days=1)).date()

    query = f"""
        SELECT * FROM scored_payments
        WHERE DATE(CONCAT(trans_year, '-', trans_month, '-', trans_day)) = '{yesterday}'
    """

    df = pd.read_sql(query, engine)

    # Sauvegarder en CSV local
    report_path = f"/opt/airflow/reports/report_{yesterday}.csv"
    df.to_csv(report_path, index=False)

    print(f"✅ Rapport généré: {report_path} avec {len(df)} lignes")

def dummy_send_email(**kwargs):
    print("📧 Ici tu pourrais envoyer le mail avec le rapport en PJ (à implémenter plus tard)")

# ======================
# 🌀 DAG
# ======================
with DAG(
    "daily_report",
    default_args=default_args,
    description="Génération d’un rapport quotidien des transactions",
    schedule_interval="0 7 * * *",  # tous les jours à 7h
    start_date=datetime(2025, 9, 25),
    catchup=False,
    tags=["fraud", "report"],
) as dag:

    task_generate_report = PythonOperator(
        task_id="generate_report",
        python_callable=generate_report,
    )

    task_send_email = PythonOperator(
        task_id="send_email",
        python_callable=dummy_send_email,
    )

    task_generate_report >> task_send_email
