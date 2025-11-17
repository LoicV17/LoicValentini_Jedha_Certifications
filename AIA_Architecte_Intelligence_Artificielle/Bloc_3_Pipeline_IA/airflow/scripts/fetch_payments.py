import os
import json
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from sqlalchemy import create_engine, text
import boto3
from io import StringIO

# ===========================
# 🔐 Variables d'environnement
# ===========================
DB_URL = os.getenv("NEONDB_URL")
PAYMENTS_API_URL = os.getenv("PAYMENTS_API_URL")
SCORING_API_URL = os.getenv("SCORING_API_URL")
S3_BUCKET = os.getenv("AIRFLOW_S3_BUCKET")

if not DB_URL or not PAYMENTS_API_URL or not SCORING_API_URL or not S3_BUCKET:
    raise ValueError("❌ Variables d'environnement manquantes")

# ===========================
# ⚙️ Connexion DB
# ===========================
engine = create_engine(DB_URL)

with engine.begin() as conn:
    # Table brute
    conn.execute(text("""
    CREATE TABLE IF NOT EXISTS raw_payments (
        id SERIAL PRIMARY KEY,
        cc_num BIGINT,
        merchant TEXT,
        category TEXT,
        amt FLOAT,
        first TEXT,
        last TEXT,
        gender TEXT,
        street TEXT,
        city TEXT,
        state TEXT,
        zip INT,
        lat FLOAT,
        long FLOAT,
        city_pop INT,
        job TEXT,
        dob TEXT,
        trans_num TEXT,
        merch_lat FLOAT,
        merch_long FLOAT,
        is_fraud INT,
        event_time BIGINT
    )
    """))

    # Table scorée
    conn.execute(text("""
    CREATE TABLE IF NOT EXISTS scored_payments (
        id SERIAL PRIMARY KEY,
        unnamed_0 BIGINT,
        category TEXT,
        amt FLOAT,
        gender TEXT,
        state TEXT,
        zip INT,
        city_pop INT,
        distance FLOAT,
        trans_year INT,
        trans_month INT,
        trans_day INT,
        trans_hour INT,
        trans_minute INT,
        trans_dayofweek INT,
        trans_week INT,
        trans_is_weekend INT,
        prediction INT,
        probability FLOAT
    )
    """))

print("✅ Tables 'raw_payments' et 'scored_payments' prêtes.")

# ===========================
# 🧹 Preprocess transaction
# ===========================
def preprocess_transaction(tx: dict) -> dict:
    try:
        ts = int(tx.get("current_time", datetime.utcnow().timestamp())) * 1000
        dt = datetime.utcfromtimestamp(ts / 1000)

        distance = np.sqrt(
            (tx.get("lat", 0) - tx.get("merch_lat", 0)) ** 2 +
            (tx.get("long", 0) - tx.get("merch_long", 0)) ** 2
        )

        return {
            "Unnamed_0": int(str(tx.get("cc_num", 0))[-9:]),
            "category": tx.get("category"),
            "amt": float(tx.get("amt", 0)),
            "gender": tx.get("gender", "M"),
            "state": tx.get("state"),
            "zip": int(tx.get("zip", 0)),
            "city_pop": int(tx.get("city_pop", 0)),
            "distance": float(distance),
            "trans_year": dt.year,
            "trans_month": dt.month,
            "trans_day": dt.day,
            "trans_hour": dt.hour,
            "trans_minute": dt.minute,
            "trans_dayofweek": dt.weekday(),
            "trans_week": dt.isocalendar()[1],
            "trans_is_weekend": 1 if dt.weekday() >= 5 else 0
        }
    except Exception as e:
        print("❌ Erreur preprocess:", e)
        return None

# ===========================
# 📦 Helper S3
# ===========================
s3 = boto3.client("s3")

def append_csv_to_s3(df_new: pd.DataFrame, bucket: str, key: str):
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        df_existing = pd.read_csv(obj["Body"])
    except s3.exceptions.NoSuchKey:
        df_existing = pd.DataFrame()

    df_final = pd.concat([df_existing, df_new], ignore_index=True)

    out_buffer = StringIO()
    df_final.to_csv(out_buffer, index=False)
    s3.put_object(Bucket=bucket, Key=key, Body=out_buffer.getvalue())

    print(f"✅ {len(df_new)} lignes ajoutées dans s3://{bucket}/{key}")

# ===========================
# 🔄 Ingestion + Scoring
# ===========================
try:
    # 1. Fetch
    resp = requests.get(PAYMENTS_API_URL, timeout=10)
    resp.raise_for_status()

    outer_json = resp.json()
    raw_json = json.loads(outer_json)

    df = pd.DataFrame(data=raw_json["data"], columns=raw_json["columns"])
    raw_tx = df.iloc[0].to_dict()

    # 2. Insert raw
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO raw_payments (
                cc_num, merchant, category, amt, first, last, gender, street, city,
                state, zip, lat, long, city_pop, job, dob, trans_num,
                merch_lat, merch_long, is_fraud, event_time
            ) VALUES (
                :cc_num, :merchant, :category, :amt, :first, :last, :gender, :street, :city,
                :state, :zip, :lat, :long, :city_pop, :job, :dob, :trans_num,
                :merch_lat, :merch_long, :is_fraud, :current_time
            )
        """), raw_tx)

    # 3. Preprocess
    clean_tx = preprocess_transaction(raw_tx)
    if not clean_tx:
        raise RuntimeError("❌ Impossible de transformer la transaction")

    # 4. Scoring
    scoring_resp = requests.post(SCORING_API_URL, json=clean_tx, timeout=10)
    scoring_resp.raise_for_status()
    scoring = scoring_resp.json()

    clean_tx["prediction"] = scoring.get("prediction", 0)
    clean_tx["probability"] = scoring.get("probability_fraud", 0.0)
    clean_tx["unnamed_0"] = clean_tx.pop("Unnamed_0")

    # 5. Insert scored
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO scored_payments (
                unnamed_0, category, amt, gender, state, zip, city_pop, distance,
                trans_year, trans_month, trans_day, trans_hour, trans_minute,
                trans_dayofweek, trans_week, trans_is_weekend,
                prediction, probability
            )
            VALUES (
                :unnamed_0, :category, :amt, :gender, :state, :zip, :city_pop, :distance,
                :trans_year, :trans_month, :trans_day, :trans_hour, :trans_minute,
                :trans_dayofweek, :trans_week, :trans_is_weekend,
                :prediction, :probability
            )
        """), clean_tx)

    print("✅ Transaction brute + scorée insérées avec succès !")

    # 6. Sauvegarde S3
    df_raw = pd.DataFrame([raw_tx])
    append_csv_to_s3(df_raw, S3_BUCKET, "fraud/raw_payments.csv")

    df_scored = pd.DataFrame([clean_tx])
    append_csv_to_s3(df_scored, S3_BUCKET, "fraud/scored_payments.csv")

except Exception as e:
    print("❌ Erreur ingestion:", e)
    raise
