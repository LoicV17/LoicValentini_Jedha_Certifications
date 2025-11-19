"""
📊 Script : generate_data_quality_report.py
🎯 Objectif : Générer un rapport Evidently avec :
    - Reference dataset = tout sauf les 24 dernières heures
    - Current dataset    = données des 24 dernières heures
"""

import os
from datetime import datetime, timedelta
import pandas as pd
from sqlalchemy import create_engine

from evidently.report import Report
from evidently.metric_preset import DataQualityPreset
from evidently.metrics import ColumnDriftMetric, DatasetSummaryMetric

# ---------------------------------------------------------------------
# 1️⃣ Chargement des données
# ---------------------------------------------------------------------
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("❌ DATABASE_URL non trouvée dans les variables d'environnement.")

engine = create_engine(DATABASE_URL)

df = pd.read_sql("SELECT * FROM reddit_scoring;", engine)
print(f"✅ Données chargées : {df.shape[0]} lignes, {df.shape[1]} colonnes")

# S’assure que la colonne timestamp existe
if "created_at" not in df.columns:
    raise RuntimeError("❌ La colonne 'created_at' est absente du dataset.")

df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")

# ---------------------------------------------------------------------
# 2️⃣ Définition des fenêtres temporelles
# ---------------------------------------------------------------------
now = datetime.utcnow()
cutoff = now - timedelta(hours=24)

df_current = df[df["created_at"] >= cutoff].copy()
df_reference = df[df["created_at"] < cutoff].copy()

print(f"📌 Current (24h)     : {len(df_current)} lignes")
print(f"📌 Reference (historique) : {len(df_reference)} lignes")

if len(df_current) < 20:
    print("⚠️ Warning : peu de données dans les 24 dernières heures, Evidently peut être moins stable.")

if len(df_reference) < 50:
    print("⚠️ Warning : peu de données historiques, drift difficile à mesurer.")

# ---------------------------------------------------------------------
# 3️⃣ Calculs techniques de qualité de données
# ---------------------------------------------------------------------
duplicate_ratio = df.duplicated(subset=["id"]).mean()
missing_ratio = df.isna().mean().mean()

emotion_cols = [
    col for col in ["joy", "anger", "sadness", "surprise", "disgust", "fear", "neutral"]
    if col in df.columns
]

for col in emotion_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

emotion_outlier_rate = (df[emotion_cols].max(axis=1) > 0.9).mean()

print("\n📊 --- DATA QUALITY STATS ---")
print(f"🔹 Duplicates ratio     : {duplicate_ratio:.3f}")
print(f"🔹 Missing value ratio  : {missing_ratio:.3f}")
print(f"🔹 Emotion outlier rate : {emotion_outlier_rate:.3f}")

# ---------------------------------------------------------------------
# 4️⃣ Rapport Evidently : Data Quality + Drift émotionnel
# ---------------------------------------------------------------------
report = Report(
    metrics=[
        DataQualityPreset(),
        DatasetSummaryMetric(),
        *[ColumnDriftMetric(col) for col in emotion_cols],
    ]
)

report.run(reference_data=df_reference, current_data=df_current)

# ---------------------------------------------------------------------
# 5️⃣ Sauvegarde des outputs
# ---------------------------------------------------------------------
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]  
# car /src/ml/... → parents[2] = racine du projet

report_dir = PROJECT_ROOT / "data" / "reports" / "evidently"
report_dir.mkdir(parents=True, exist_ok=True)

os.makedirs(report_dir, exist_ok=True)

report_path = os.path.join(report_dir, f"reddit_data_quality_{timestamp}.html")
report.save_html(report_path)

print(f"✅ Rapport Evidently sauvegardé : {report_path}")

summary_df = pd.DataFrame([{
    "timestamp": timestamp,
    "rows_total": len(df),
    "rows_reference": len(df_reference),
    "rows_current": len(df_current),
    "duplicates_ratio": duplicate_ratio,
    "missing_ratio": missing_ratio,
    "emotion_outlier_rate": emotion_outlier_rate,
}])

summary_path = os.path.join(report_dir, f"reddit_data_quality_summary_{timestamp}.csv")
summary_df.to_csv(summary_path, index=False)

print(f"✅ Fichier résumé créé : {summary_path}")
print("🎉 Analyse Evidently terminée.")
