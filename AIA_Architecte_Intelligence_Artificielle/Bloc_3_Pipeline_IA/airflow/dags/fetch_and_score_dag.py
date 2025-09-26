# dags/fetch_and_score_dag.py
from airflow import DAG
from airflow.operators.bash import BashOperator  # type: ignore
from datetime import datetime, timedelta

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    "fetch_and_score",
    default_args=default_args,
    description="Pipeline d’ingestion + scoring en continu",
    schedule_interval="* * * * *",  # toutes les minutes
    start_date=datetime(2025, 9, 25),
    catchup=False,
    tags=["fraud", "streaming"],
) as dag:

    run_fetch = BashOperator(
        task_id="fetch_and_score_task",
        bash_command="python /opt/airflow/scripts/fetch_payments.py"
,
    )
