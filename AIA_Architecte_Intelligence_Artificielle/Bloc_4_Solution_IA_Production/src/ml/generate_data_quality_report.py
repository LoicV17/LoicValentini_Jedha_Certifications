"""
📊 Script : generate_data_quality_report.py
🎯 Objectif : Générer un rapport Evidently orienté Data Quality sur les scores émotionnels Reddit
"""

import os
from datetime import datetime
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
    raise RuntimeError("❌ DATABASE_URL non trouvée.")

engine = create_engine(DATABASE_URL)
df = pd.read_sql("SELECT * FROM reddit_scoring;", engine)
print(f"✅ Données chargées : {df.shape[0]} lignes, {df.shape[1]} colonnes")

# ---------------------------------------------------------------------
# 2️⃣ Calculs techniques de qualité de données
# ---------------------------------------------------------------------
duplicate_ratio = df.duplicated(subset=["id"]).mean()
missing_ratio = df.isna().mean().mean()

emotion_cols = [c for c in ["joy","anger","sadness","surprise","disgust","fear","neutral"] if c in df.columns]
emotion_outlier_rate = (df[emotion_cols].max(axis=1) > 0.9).mean()

print("📊 --- DATA QUALITY STATS ---")
print(f"🔹 Duplicates ratio     : {duplicate_ratio:.3f}")
print(f"🔹 Missing value ratio  : {missing_ratio:.3f}")
print(f"🔹 Emotion outlier rate : {emotion_outlier_rate:.3f}")

# ---------------------------------------------------------------------
# 3️⃣ Rapport Evidently simplifié (orienté émotion)
# ---------------------------------------------------------------------
for col in emotion_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

report = Report(
    metrics=[
        DataQualityPreset(),
        DatasetSummaryMetric(),
        *[ColumnDriftMetric(col) for col in emotion_cols],
    ]
)

report.run(reference_data=df, current_data=df)

# ---------------------------------------------------------------------
# 4️⃣ Sauvegarde du rapport et résumé
# ---------------------------------------------------------------------
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
report_dir = "/opt/airflow/data/reports/evidently"
os.makedirs(report_dir, exist_ok=True)

report_path = os.path.join(report_dir, f"reddit_data_quality_{timestamp}.html")
report.save_html(report_path)
print(f"✅ Rapport Evidently sauvegardé : {report_path}")

summary_df = pd.DataFrame([{
    "timestamp": timestamp,
    "rows": len(df),
    "duplicates_ratio": duplicate_ratio,
    "missing_ratio": missing_ratio,
    "emotion_outlier_rate": emotion_outlier_rate,
}])
summary_path = os.path.join(report_dir, f"reddit_data_quality_summary_{timestamp}.csv")
summary_df.to_csv(summary_path, index=False)
print(f"✅ Fichier résumé créé : {summary_path}")
