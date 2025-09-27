# dags/full_report_refresh_dag.py
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import boto3
import pandas as pd
from io import StringIO

import os

BUCKET = os.getenv("AIRFLOW_S3_BUCKET")

def refresh_full_report():
    s3 = boto3.client("s3")

    # Lire le CSV brut
    obj = s3.get_object(Bucket=BUCKET, Key="fraud/scored_payments.csv")
    df = pd.read_csv(obj["Body"])

    # Sauvegarder en Parquet optimisé
    parquet_key = "reports/full/scored_payments.parquet"
    out_buffer = df.to_parquet(index=False)
    s3.put_object(Bucket=BUCKET, Key=parquet_key, Body=out_buffer)

    # Créer un flag READY
    s3.put_object(Bucket=BUCKET, Key="reports/full/READY", Body=b"ok")

    print(f"✅ Rapport complet mis à jour : s3://{BUCKET}/{parquet_key}")


default_args = {
    "owner": "airflow",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="full_report_refresh",
    default_args=default_args,
    description="Rafraîchit le dataset complet pour Streamlit",
    schedule_interval="0 * * * *",  # toutes les minutes
    start_date=datetime(2025, 9, 26),
    catchup=False,
    tags=["fraud", "reports"],
) as dag:

    task_refresh = PythonOperator(
        task_id="refresh_full_report",
        python_callable=refresh_full_report,
    )
