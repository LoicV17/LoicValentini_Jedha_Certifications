# src/ml/train_topics_from_neondb.py

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]   # ton dossier "Bloc_4_Solution..."
SRC_DIR = ROOT / "src"
sys.path.append(str(SRC_DIR))

import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
from ml.topic_clustering import train_topic_clusters  # chemin à adapter selon ton package

load_dotenv()

DB_URL = os.getenv("DATABASE_URL")  # déjà défini pour ton projet
engine = create_engine(DB_URL)


def main():
    query = """
    SELECT
        id,
        title,
        selftext,
        COALESCE(selftext, '') || ' ' || COALESCE(title, '') AS text
    FROM reddit_cleaned
    WHERE created_utc >= NOW() - INTERVAL '30 days'
    """
    df = pd.read_sql(query, engine)

    print(f"[TrainTopics] Loaded {len(df)} posts from NeonDB.")
    train_topic_clusters(df, text_col="text", n_clusters=8)


if __name__ == "__main__":
    main()
