# src/ml/score_emotions.py

import os
import pandas as pd
from sqlalchemy import create_engine
from transformers import pipeline
from dotenv import load_dotenv
from tqdm import tqdm
from datetime import datetime
from pathlib import Path

# --- Charger variables d'environnement ---
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise RuntimeError("❌ DATABASE_URL non trouvée. Vérifie ton .env ou docker-compose.yml")

# --- Connexion à NeonDB ---
engine = create_engine(DATABASE_URL)

# --- Charger les nouveaux posts ---
query = """
SELECT id, title, selftext
FROM reddit_cleaned
WHERE id NOT IN (SELECT id FROM reddit_scoring);
"""
df = pd.read_sql(query, engine)

if df.empty:
    print("✅ Aucun nouveau post à scorer.")
    exit(0)

print(f"🔍 {len(df)} posts à analyser...")

# --- Charger le modèle ---
emotion_classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    return_all_scores=True
)

# --- Application du modèle ---
records = []
for _, row in tqdm(df.iterrows(), total=len(df)):
    text = f"{row['title']} {row['selftext'] or ''}".strip()
    if not text:
        continue

    # Tronquer les textes longs pour éviter les erreurs de séquence (>512 tokens)
    if len(text.split()) > 200:  # environ 512 tokens équivalents
        text = " ".join(text.split()[:200])

    try:
        preds = emotion_classifier(text, truncation=True, max_length=512)[0]
    except Exception as e:
        print(f"⚠️ Erreur sur un texte trop long ou invalide : {e}")
        continue

    emotion_scores = {p['label']: p['score'] for p in preds}
    main_emotion = max(emotion_scores, key=emotion_scores.get)

    record = {
        "id": row["id"],
        **emotion_scores,
        "main_emotion": main_emotion,
    }
    records.append(record)

# --- Sauvegarde dans la base ---
df_out = pd.DataFrame(records)
df_out.to_sql("reddit_scoring", engine, if_exists="append", index=False)

# --- Sauvegarde CSV ---
output_dir = Path("/opt/airflow/data/scored")
output_dir.mkdir(parents=True, exist_ok=True)

output_path = output_dir / f"reddit_scoring_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
df_out.to_csv(output_path, index=False)

print(f"✅ {len(df_out)} posts analysés et sauvegardés dans la base et {output_path}")
