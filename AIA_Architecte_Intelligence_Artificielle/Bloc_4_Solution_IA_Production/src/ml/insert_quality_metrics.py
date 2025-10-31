"""
📥 Script : insert_quality_metrics.py
🎯 Objectif : Insérer l’historique des métriques de Data Quality dans NeonDB
"""

import os
import pandas as pd
from sqlalchemy import create_engine, text

# ---------------------------------------------------------------------
# 1️⃣ Chargement du dernier fichier résumé
# ---------------------------------------------------------------------
report_dir = "/opt/airflow/data/reports"

csv_files = sorted(
    [f for f in os.listdir(report_dir) if f.startswith("reddit_data_quality_summary_")],
    reverse=True
)
if not csv_files:
    raise FileNotFoundError("❌ Aucun fichier summary trouvé dans /opt/airflow/data/reports")

latest_csv = os.path.join(report_dir, csv_files[0])
summary_df = pd.read_csv(latest_csv)

print(f"✅ Chargement du fichier : {latest_csv}")
print(summary_df.head())

# ---------------------------------------------------------------------
# 2️⃣ Normalisation des colonnes attendues
# ---------------------------------------------------------------------
expected_cols = ["timestamp", "rows", "duplicates_ratio", "missing_ratio", "emotion_outlier_rate"]
summary_df = summary_df[[col for col in expected_cols if col in summary_df.columns]]

# ---------------------------------------------------------------------
# 3️⃣ Connexion à la base NeonDB
# ---------------------------------------------------------------------
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("❌ DATABASE_URL non trouvée.")

engine = create_engine(DATABASE_URL)

# ---------------------------------------------------------------------
# 4️⃣ Création de la table si nécessaire (corrigé pour SQLAlchemy ≥ 2.0)
# ---------------------------------------------------------------------
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

# ✅ Le context manager "engine.begin()" gère le commit automatiquement
with engine.begin() as conn:
    conn.execute(create_table_query)

# ---------------------------------------------------------------------
# 5️⃣ Insertion des nouvelles métriques
# ---------------------------------------------------------------------
summary_df.to_sql("reddit_quality_history", con=engine, if_exists="append", index=False)
print("✅ Historique de qualité inséré dans NeonDB avec succès !")
