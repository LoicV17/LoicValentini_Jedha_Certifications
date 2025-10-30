import os
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from pathlib import Path

# --- 1️⃣ Charger la variable d’environnement ---
load_dotenv()
DB_URL = os.getenv("DATABASE_URL")

if not DB_URL:
    raise RuntimeError("❌ DATABASE_URL non trouvée. Vérifie ton .env ou ton docker-compose.yml")

# --- 2️⃣ Connexion à NeonDB ---
engine = create_engine(DB_URL)

# --- 3️⃣ Cibler le dernier CSV nettoyé ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
latest_file = sorted(PROCESSED_DIR.glob("reddit_cleaned_*.csv"), key=lambda f: f.stat().st_mtime, reverse=True)[0]

print(f"📂 Chargement du fichier : {latest_file}")

df = pd.read_csv(latest_file)

# --- 4️⃣ Récupérer les IDs déjà présents dans la DB ---
with engine.connect() as conn:
    existing_ids = pd.read_sql(text("SELECT id FROM reddit_cleaned"), conn)["id"].tolist()

# --- 5️⃣ Filtrer les doublons ---
df_new = df[~df["id"].isin(existing_ids)]
print(f"🧹 {len(df_new)} nouveaux posts à insérer sur {len(df)} totaux.")

if not df_new.empty:
    df_new.to_sql("reddit_cleaned", engine, if_exists="append", index=False)
    print(f"✅ {len(df_new)} lignes insérées dans la table 'reddit_cleaned'")
else:
    print("ℹ️ Aucun nouveau post à insérer.")
