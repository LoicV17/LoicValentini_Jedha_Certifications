# src/ml/cluster_topics.py

import pandas as pd
from pathlib import Path
from ml.topic_clustering import assign_topic_clusters  # utilise le module que je t’ai généré
import logging

# Répertoires (à adapter selon ta structure)
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
CLEAN_FILE = DATA_DIR / "reddit_clean.parquet"
CLUSTERED_FILE = DATA_DIR / "reddit_clustered.parquet"


def main():
    logging.info("📌 Chargement des données nettoyées...")
    if not CLEAN_FILE.exists():
        raise FileNotFoundError(f"❌ Fichier nettoyé introuvable : {CLEAN_FILE}")

    df = pd.read_parquet(CLEAN_FILE)

    logging.info(f"📄 {len(df)} posts chargés pour clustering.")

    # Construire une colonne texte cohérente (title + selftext)
    df["text"] = df["title"].fillna("") + " " + df["selftext"].fillna("")

    logging.info("🔮 Attribution des clusters thématiques...")
    df_clustered = assign_topic_clusters(df, text_col="text")

    logging.info("💾 Sauvegarde du fichier clusterisé...")
    df_clustered.to_parquet(CLUSTERED_FILE, index=False)

    logging.info(f"✅ Clusterisation terminée → {CLUSTERED_FILE}")


if __name__ == "__main__":
    main()
