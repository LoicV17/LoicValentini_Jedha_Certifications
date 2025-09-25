# test_neondb.py
from pathlib import Path
from dotenv import load_dotenv
import os
from sqlalchemy import create_engine, text

# =============================
# 🔐 Charger secrets (.env)
# =============================
env_path = Path(__file__).parent / "secrets" / ".env"
load_dotenv(dotenv_path=env_path)

DB_URL = os.getenv("NEONDB_URL")

if not DB_URL:
    raise ValueError("❌ NEONDB_URL introuvable dans le fichier .env")

print("✅ URL de connexion récupérée depuis .env")

# =============================
# 🔗 Connexion NeonDB
# =============================
try:
    engine = create_engine(DB_URL)
    with engine.connect() as conn:
        result = conn.execute(text("SELECT version();"))
        print("✅ Connexion réussie à NeonDB")
        print("PostgreSQL version:", result.scalar())
except Exception as e:
    print("❌ Erreur de connexion NeonDB:", e)
