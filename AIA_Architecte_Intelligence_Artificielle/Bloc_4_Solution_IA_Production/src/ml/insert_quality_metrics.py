"""
📥 Script : insert_quality_metrics.py
🎯 Objectif : Insérer l’historique des métriques de Data Quality dans NeonDB
"""

import os
import pandas as pd
from pathlib import Path
from sqlalchemy import create_engine, text
import glob

# ------------------------------------------------------------------------------
# 1️⃣ Localisation du dossier Evidently
# ------------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # remonte jusqu’à Bloc45/
EVIDENTLY_DIR = PROJECT_ROOT / "data" / "reports" / "evidently"

print(f"📁 Dossier Evidently : {EVIDENTLY_DIR}")

if not EVIDENTLY_DIR.exists():
    raise FileNotFoundError(f"❌ Le dossier Evidently n’existe pas : {EVIDENTLY_DIR}")

# ------------------------------------------------------------------------------
# 2️⃣ Récupération du dernier fichier summary
# ------------------------------------------------------------------------------
summary_files = sorted(
    glob.glob(str(EVIDENTLY_DIR / "reddit_data_quality_summary_*.csv"))
)

if not summary_files:
    raise FileNotFoundError(f"❌ Aucun fichier summary trouvé dans {EVIDENTLY_DIR}")

latest_summary = summary_files[-1]
print(f"📄 Fichier summary détecté : {latest_summary}")

summary_df = pd.read_csv(latest_summary)
print("🔍 Aperçu des métriques :")
print(summary_df.head())

# ------------------------------------------------------------------------------
# 3️⃣ Normalisation des colonnes attendues
# ------------------------------------------------------------------------------
expected_cols = [
    "timestamp",
    "rows",
    "duplicates_ratio",
    "missing_ratio",
    "emotion_outlier_rate"
]

summary_df = summary_df[[c for c in expected_cols if c in summary_df.columns]]

# ------------------------------------------------------------------------------
# 4️⃣ Connexion NeonDB
# ------------------------------------------------------------------------------
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise RuntimeError("❌ DATABASE_URL non trouvée dans les variables d’environnement.")

engine = create_engine(DATABASE_URL)

# ------------------------------------------------------------------------------
# 5️⃣ Création table si nécessaire
# ------------------------------------------------------------------------------
create_table_query = text("""
CREATE TABLE IF NOT EXISTS reddit_quality_history (
    id SERIAL PRIMARY KEY,
    timestamp TEXT,
    rows INTEGER,
    duplicates_ratio FLOAT,
    missing_ratio FLOAT,
    emotion_outlier_rate FLOAT
);
""")

with engine.begin() as conn:
    conn.execute(create_table_query)

# ------------------------------------------------------------------------------
# 6️⃣ Insertion dans NeonDB
# ------------------------------------------------------------------------------
summary_df.to_sql(
    "reddit_quality_history",
    con=engine,
    if_exists="append",
    index=False
)

print("✅ Données ajoutées avec succès dans reddit_quality_history !")
