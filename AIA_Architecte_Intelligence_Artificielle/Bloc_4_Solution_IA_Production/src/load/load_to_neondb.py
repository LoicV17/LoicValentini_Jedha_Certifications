import os
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from pathlib import Path

# Charger env
load_dotenv()
DB_URL = os.getenv("DATABASE_URL")

if not DB_URL:
    raise RuntimeError("❌ DATABASE_URL non trouvée. Vérifie ton .env ou ton docker-compose.yml")

engine = create_engine(DB_URL)

# Dernier fichier nettoyé
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

latest_file = sorted(
    PROCESSED_DIR.glob("reddit_cleaned_*.csv"),
    key=lambda f: f.stat().st_mtime,
    reverse=True
)[0]

print(f"📂 Chargement du fichier : {latest_file}")

df = pd.read_csv(latest_file)

# IDs existants
with engine.connect() as conn:
    try:
        existing_ids = pd.read_sql(text("SELECT id FROM reddit_cleaned"), conn)["id"].tolist()
    except Exception:
        existing_ids = []

# Filtrer doublons
df_new = df[~df["id"].isin(existing_ids)]
print(f"🧹 {len(df_new)} nouveaux posts.")

if not df_new.empty:
    df_new.to_sql("reddit_cleaned", engine, if_exists="append", index=False)
    print("✅ Insertion en base OK")
else:
    print("ℹ️ Aucun nouveau post.")
