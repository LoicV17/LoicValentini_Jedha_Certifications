import os
import pandas as pd
import joblib
from sqlalchemy import create_engine, text
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()
DB_URL = os.getenv("DATABASE_URL")
if not DB_URL:
    raise RuntimeError("❌ DATABASE_URL manquant")

engine = create_engine(DB_URL)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "src" / "models" / "topic_clustering" / "cluster_model.joblib"

print("➡️ Chargement modèle :", MODEL_PATH)
cluster_pack = joblib.load(MODEL_PATH)

# --- Extraction robuste du modèle ---
def extract_kmeans(obj):
    if isinstance(obj, dict):
        # notre cas actuel
        if "kmeans" in obj:
            return obj["kmeans"]

        # fallback pour future compatibilité
        for key, val in obj.items():
            if hasattr(val, "predict"):
                return val

        raise KeyError(f"❌ Aucun modèle avec .predict() trouvé. Clés : {list(obj.keys())}")

    if hasattr(obj, "predict"):
        return obj

    raise TypeError("❌ Format du fichier modèle non reconnu")

kmeans = extract_kmeans(cluster_pack)
print("✔ Modèle extrait :", type(kmeans))

# --- Charger posts non clusterisés ---
query = """
    SELECT id, title, selftext,
           COALESCE(selftext, '') || ' ' || COALESCE(title, '') AS text
    FROM reddit_cleaned
    WHERE topic_cluster IS NULL
      AND created_utc >= NOW() - INTERVAL '30 days'
"""
df = pd.read_sql(query, engine)
print(f"📥 {len(df)} posts à prédire")

if df.empty:
    print("ℹ️ Aucun post à clusteriser")
    exit(0)

# --- Encoder les textes ---
model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
embeddings = model.encode(df["text"].tolist(), batch_size=32, show_progress_bar=True)

# --- Prédiction ---
df["topic_cluster"] = kmeans.predict(embeddings)

print(df.head())

# --- Insert / Update ---
with engine.begin() as conn:
    for _, row in df.iterrows():
        conn.execute(
            text("""
                UPDATE reddit_cleaned
                SET topic_cluster = :topic
                WHERE id = :id
            """),
            {"topic": int(row["topic_cluster"]), "id": row["id"]}
        )

print("✅ Mise à jour terminée dans la DB")
