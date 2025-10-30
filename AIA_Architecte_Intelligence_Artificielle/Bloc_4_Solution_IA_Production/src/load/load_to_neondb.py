import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
from pathlib import Path

# --- 1️⃣ Charger la variable d’environnement ---
# (utile si tu exécutes le script manuellement hors Docker)
load_dotenv()
DB_URL = os.getenv("DATABASE_URL")

if not DB_URL:
    raise RuntimeError("❌ DATABASE_URL non trouvée. Vérifie ton .env ou ton docker-compose.yml")

# --- 2️⃣ Connexion à NeonDB ---
engine = create_engine(DB_URL)

# --- 3️⃣ Cibler le dernier CSV nettoyé ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

latest_file = sorted(PROCESSED_DIR.glob("reddit_cleaned_*.csv"),
                     key=lambda f: f.stat().st_mtime,
                     reverse=True)[0]

print(f"📂 Chargement du fichier : {latest_file}")

# --- 4️⃣ Charger et insérer ---
df = pd.read_csv(latest_file)
df.to_sql("reddit_cleaned", engine, if_exists="append", index=False)

print(f"✅ {len(df)} lignes insérées dans la table 'reddit_cleaned'")
