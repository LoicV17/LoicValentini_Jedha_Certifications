from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
from pathlib import Path
import os
import boto3
from dotenv import load_dotenv
from urllib.parse import urlparse

# ===========================
# 🔐 Charger secrets (.env)
# ===========================
env_path = Path(__file__).parent.parent / "secrets" / ".env"
load_dotenv(dotenv_path=env_path)

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "eu-west-3")

# URL S3 exacte du modèle OneHot à mettre dans ton .env
S3_URL_MODEL = os.getenv("S3_URL_MODEL")  # ex: s3://fraud-detection-loicvalentini/models/xgboost_fraud_model_onehot.pkl

# ===========================
# 📂 Dossier temporaire accessible en écriture (HF autorise /tmp)
# ===========================

models_dir = Path("/tmp/models_s3")
models_dir.mkdir(parents=True, exist_ok=True)


def download_from_s3_url(s3_url, local_filename):
    """Télécharge un fichier S3 donné par son URL complète s3://..."""
    if not s3_url.startswith("s3://"):
        raise ValueError(f"URL S3 invalide : {s3_url}")

    parsed = urlparse(s3_url, allow_fragments=False)
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")  # enlever le premier "/"

    s3 = boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_REGION,
    )

    local_path = models_dir / local_filename
    print(f"⬇️ Downloading {s3_url} -> {local_path}")
    s3.download_file(bucket, key, str(local_path))
    return local_path

# ===========================
# 📥 Charger modèle OneHot
# ===========================
try:
    model_path = download_from_s3_url(S3_URL_MODEL, "xgboost_fraud_model_onehot.pkl")
    model = joblib.load(model_path)
    print("✅ Modèle (pipeline OneHot) chargé depuis S3")
except Exception as e:
    print("❌ Erreur lors du chargement depuis S3:", e)
    model = None

# ===========================
# 🚀 API FastAPI
# ===========================
app = FastAPI(title="Fraud Detection API (OneHot from S3)")

class Transaction(BaseModel):
    Unnamed_0: int
    category: str
    amt: float
    gender: str
    state: str
    zip: int
    city_pop: int
    distance: float
    trans_year: int
    trans_month: int
    trans_day: int
    trans_hour: int
    trans_minute: int
    trans_dayofweek: int
    trans_week: int
    trans_is_weekend: int

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/predict")
def predict(transaction: Transaction):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    data = transaction.dict()

    # ⚠️ Gender reste numérique comme à l'entraînement (0/1)
    gender_map = {"M": 1, "F": 0}
    data["gender"] = gender_map.get(data["gender"].upper(), 0)

    # ✅ Supprimer colonne inutile
    input_df = pd.DataFrame([data]).drop(columns=["Unnamed_0"])

    # ✅ Le pipeline OneHot fait le reste
    proba = float(model.predict_proba(input_df)[0, 1])
    pred = int(model.predict(input_df)[0])

    return {"prediction": pred, "probability_fraud": proba}
