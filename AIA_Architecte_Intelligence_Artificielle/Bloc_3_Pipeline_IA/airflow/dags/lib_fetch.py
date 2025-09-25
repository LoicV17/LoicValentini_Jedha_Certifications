import os
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from pathlib import Path
from datetime import datetime

# Charger secrets
env_path = Path("/opt/airflow/secrets/.env")
load_dotenv(dotenv_path=env_path)

DB_URL = os.getenv("NEONDB_URL")
engine = create_engine(DB_URL)


def run_fetch_and_score():
    """
    Appelle directement ton script fetch_payments.py
    """
    os.system("python /opt/airflow/fetch_payments.py")


def generate_daily_report():
    """
    Générer un rapport CSV des transactions scorées de la veille
    """
    query = """
        SELECT *
        FROM scored_payments
        WHERE trans_year = EXTRACT(YEAR FROM CURRENT_DATE - INTERVAL '1 day')
          AND trans_month = EXTRACT(MONTH FROM CURRENT_DATE - INTERVAL '1 day')
          AND trans_day = EXTRACT(DAY FROM CURRENT_DATE - INTERVAL '1 day');
    """

    df = pd.read_sql(query, engine)

    report_path = f"/opt/airflow/reports/daily_report_{datetime.now().date()}.csv"
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    df.to_csv(report_path, index=False)

    print(f"✅ Rapport généré : {report_path}")
    return report_path
