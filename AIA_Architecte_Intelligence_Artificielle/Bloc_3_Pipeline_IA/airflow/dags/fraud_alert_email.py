# dags/fraud_alert_email.py

from airflow.decorators import dag, task
from datetime import timedelta
import pendulum
import boto3, os, io
import pandas as pd
from airflow.operators.email import EmailOperator

# =========
# Variables
# =========
BUCKET = os.getenv("AIRFLOW_S3_BUCKET")
PARQUET_KEY = "reports/full/scored_payments.parquet"
EMAIL_RECIPIENTS = [
    e.strip()
    for e in os.getenv("REPORT_EMAIL_TO", "").split(",")
    if e.strip()
]


@dag(
    dag_id="fraud_alert_email",
    schedule_interval="* * * * *",   # toutes les minutes
    start_date=pendulum.datetime(2025, 9, 25, tz="Europe/Paris"),
    catchup=False,
    default_args={"owner": "airflow", "retries": 0},
    tags=["fraud", "alerts", "email"],
)
def fraud_alert_email():

    @task
    def find_new_frauds() -> str:
        """
        Cherche des fraudes sur une fenêtre glissante [now-2min, now).
        Retourne du HTML si fraude(s), sinon chaîne vide.
        """
        tz = pendulum.timezone("Europe/Paris")
        now = pendulum.now(tz)
        window_start = now - timedelta(minutes=2)  # tolérance anti-lag
        window_end = now

        # Charger parquet depuis S3
        s3 = boto3.client("s3")
        obj = s3.get_object(Bucket=BUCKET, Key=PARQUET_KEY)
        df = pd.read_parquet(io.BytesIO(obj["Body"].read()))

        # Reconstruire event_time comme dans Streamlit si nécessaire
        if {"trans_year", "trans_month", "trans_day", "trans_hour"}.issubset(df.columns):
            df["event_time"] = pd.to_datetime(
                df["trans_year"].astype(str) + "-" +
                df["trans_month"].astype(str).str.zfill(2) + "-" +
                df["trans_day"].astype(str).str.zfill(2) + " " +
                df["trans_hour"].astype(str).str.zfill(2) + ":" +
                df.get("trans_minute", 0).astype(str).str.zfill(2),
                errors="coerce",
            )
        else:
            if "event_time" in df.columns:
                df["event_time"] = pd.to_datetime(df["event_time"], errors="coerce")
            else:
                df["event_time"] = pd.NaT

        # Debug : bornes et stats temps
        try:
            ev_min = df["event_time"].min()
            ev_max = df["event_time"].max()
            print(
                f"[DEBUG] window: {window_start.to_datetime_string()} → "
                f"{window_end.to_datetime_string()} (Europe/Paris)"
            )
            print(f"[DEBUG] event_time min={ev_min} max={ev_max} rows={len(df)}")
        except Exception as e:
            print(f"[DEBUG] time stats error: {e}")

        # Comparaison en naive (comme nos event_time)
        start_naive = window_start.naive()
        end_naive = window_end.naive()

        mask = (
            (df.get("prediction", 0) == 1)
            & (df["event_time"] >= start_naive)
            & (df["event_time"] < end_naive)
        )
        frauds = df.loc[mask].copy()

        print(f"[DEBUG] frauds_in_window={len(frauds)}")

        if frauds.empty:
            return ""

        cols = [
            c
            for c in ["event_time", "amt", "merchant", "category", "state", "probability"]
            if c in frauds.columns
        ]
        frauds = frauds.sort_values("event_time", ascending=False)

        MAX_ROWS = 200  # anti-spam
        table_html = frauds[cols].head(MAX_ROWS).to_html(index=False, border=0)

        total = len(frauds)
        total_amt = float(frauds["amt"].sum()) if "amt" in frauds.columns else 0.0

        html = f"""
        <h2>🚨 Alerte fraude détectée</h2>
        <p>Fenêtre : {window_start.to_datetime_string()} → {window_end.to_datetime_string()} (Europe/Paris)</p>
        <div>
            <b>Nombre de fraudes :</b> {total:,} &nbsp;|&nbsp;
            <b>Montant total fraudé (€) :</b> <span style="color:red;">{total_amt:,.0f}</span>
        </div>
        <h3>Détails</h3>
        {table_html}
        """
        if total > MAX_ROWS:
            html += f"<p style='color:#666;'>… {total - MAX_ROWS} lignes supplémentaires non affichées.</p>"

        return html

    @task
    def send_alert(html_body: str):
        if not html_body:
            print("Aucune fraude détectée — pas d'email envoyé.")
            return
        if not EMAIL_RECIPIENTS:
            print("⚠️ Pas de destinataires (REPORT_EMAIL_TO vide).")
            return

        # On utilise EmailOperator à l'intérieur du taskflow
        EmailOperator(
            task_id="send_fraud_alert_email",
            to=EMAIL_RECIPIENTS,
            subject="🚨 Alerte fraude — nouvelles transactions détectées",
            html_content=html_body,
            conn_id="smtp_gmail_basic",  # même conn_id que ton ancien SmtpHook
        ).execute(context={})

        print("✅ Email d’alerte envoyé.")

    send_alert(find_new_frauds())


dag = fraud_alert_email()
