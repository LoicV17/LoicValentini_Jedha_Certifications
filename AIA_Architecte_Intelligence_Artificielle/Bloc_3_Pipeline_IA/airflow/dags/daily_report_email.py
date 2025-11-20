# dags/daily_report_email.py

from airflow.decorators import dag, task
from airflow.operators.email import EmailOperator
from datetime import timedelta
import pendulum
import boto3, os, io
import pandas as pd

# Variables
BUCKET = os.getenv("AIRFLOW_S3_BUCKET")
PARQUET_KEY = "reports/full/scored_payments.parquet"
EMAIL_RECIPIENTS = os.getenv("REPORT_EMAIL_TO", "").split(",")

@dag(
    dag_id="daily_report_email",
    schedule_interval="0 7 * * *",  # Tous les jours à 07:00
    start_date=pendulum.datetime(2025, 9, 25, tz="Europe/Paris"),
    catchup=False,
    default_args={
        "owner": "airflow",
        "retries": 1,
        "retry_delay": timedelta(minutes=5),
    },
    tags=["fraud", "reports", "daily", "email"],
)
def daily_report_email():

    @task
    def build_daily_html():
        # Charger depuis S3
        s3 = boto3.client("s3")
        obj = s3.get_object(Bucket=BUCKET, Key=PARQUET_KEY)
        df = pd.read_parquet(io.BytesIO(obj["Body"].read()))

        # Reconstruire event_time si nécessaire
        if {"trans_year", "trans_month", "trans_day", "trans_hour"}.issubset(df.columns):
            df["event_time"] = pd.to_datetime(
                df["trans_year"].astype(str) + "-" +
                df["trans_month"].astype(str).str.zfill(2) + "-" +
                df["trans_day"].astype(str).str.zfill(2) + " " +
                df["trans_hour"].astype(str).str.zfill(2) + ":" +
                df.get("trans_minute", 0).astype(str).str.zfill(2),
                errors="coerce"
            )
        else:
            df["event_time"] = pd.NaT

        # Filtre 24h
        now = pendulum.now("Europe/Paris")
        start = now - timedelta(days=1)
        df = df[(df["event_time"] >= start.naive()) & (df["event_time"] < now.naive())]

        if df.empty:
            return "<h3>Rapport quotidien – aucune transaction sur les 24 dernières heures.</h3>"

        # KPIs
        total_tx = len(df)
        total_amt = df["amt"].sum()
        frauds = df[df["prediction"] == 1]
        fraud_count = len(frauds)
        fraud_rate = 100 * fraud_count / total_tx if total_tx else 0
        fraud_amt = frauds["amt"].sum()

        # Tableau
        cols = [c for c in ["event_time","amt","merchant","category","state","probability"] if c in frauds.columns]
        fraude_table = frauds[cols].sort_values("event_time", ascending=False).to_html(index=False, border=0)

        html = f"""
        <h2>🕵️ Rapport Fraude – dernières 24h</h2>
        <p>Période : {start.to_datetime_string()} → {now.to_datetime_string()} (Europe/Paris)</p>

        <div>
            <b>Transactions :</b> {total_tx:,} |
            <b>Fraudes :</b> {fraud_count:,} |
            <b>Taux :</b> {fraud_rate:.2f}% |
            <b>Montant (€) :</b> {total_amt:,.0f} |
            <b>Fraudé (€) :</b> <span style="color:red;">{fraud_amt:,.0f}</span>
        </div>

        <h3>📂 Détails des fraudes détectées</h3>
        {fraude_table}
        """
        return html

    @task
    def store_html_in_s3(html_body: str):
        s3 = boto3.client("s3")
        now = pendulum.now("Europe/Paris")
        key = f"reports/daily/report_{now.format('YYYYMMDD')}.html"

        s3.put_object(
            Bucket=BUCKET,
            Key=key,
            Body=html_body.encode("utf-8"),
            ContentType="text/html"
        )
        print(f"✅ Rapport stocké : s3://{BUCKET}/{key}")

        return html_body

    # EmailOperator prend directement le HTML
    send_email = EmailOperator(
        task_id="send_email_report",
        to=EMAIL_RECIPIENTS,
        subject="Rapport Fraude – dernières 24h",
        html_content="{{ ti.xcom_pull(task_ids='store_html_in_s3') }}",
        conn_id="gmail_smtp"
    )

    html = build_daily_html()
    html_stored = store_html_in_s3(html)
    html_stored >> send_email

dag = daily_report_email()
