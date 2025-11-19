# src/ml/topic_clustering.py

import os
from pathlib import Path
import joblib
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import MiniBatchKMeans

BASE_DIR = Path(__file__).resolve().parents[1]  # .../Bloc_4_Solution_IA_Production
MODEL_DIR = BASE_DIR / "models" / "topic_clustering"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

CLUSTER_MODEL_PATH = MODEL_DIR / "cluster_model.joblib"

# On choisit un petit modèle rapide et efficace
EMBEDDER_NAME = "sentence-transformers/paraphrase-MiniLM-L6-v2"


def load_embedder():
    return SentenceTransformer(EMBEDDER_NAME)


def train_topic_clusters(
    df: pd.DataFrame,
    text_col: str = "text",
    n_clusters: int = 8,
    batch_size: int = 512,
):
    """
    Entraîne un modèle de clustering thématique sur les posts Reddit.
    df doit contenir une colonne 'text' (ou celle indiquée).
    """

    df = df.dropna(subset=[text_col]).copy()
    df = df[df[text_col].str.strip() != ""]

    embedder = load_embedder()
    texts = df[text_col].tolist()

    print(f"[TopicClustering] Encoding {len(texts)} posts...")
    embeddings = embedder.encode(texts, batch_size=64, show_progress_bar=True)

    print(f"[TopicClustering] Fitting MiniBatchKMeans (k={n_clusters})...")
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=42,
        batch_size=batch_size,
        verbose=1,
    )
    kmeans.fit(embeddings)

    joblib.dump({"kmeans": kmeans}, CLUSTER_MODEL_PATH)
    print(f"[TopicClustering] Model saved at {CLUSTER_MODEL_PATH}")


def assign_topic_clusters(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """
    Charge le modèle de clustering et assigne un cluster à chaque post.
    Retourne un df avec une colonne 'topic_cluster'.
    """

    if not CLUSTER_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Cluster model not found at {CLUSTER_MODEL_PATH}. "
            "Train it first with train_topic_clusters()."
        )

    model_bundle = joblib.load(CLUSTER_MODEL_PATH)
    kmeans = model_bundle["kmeans"]

    df = df.copy()
    df = df.dropna(subset=[text_col])
    df = df[df[text_col].str.strip() != ""]

    embedder = load_embedder()
    texts = df[text_col].tolist()
    embeddings = embedder.encode(texts, batch_size=64, show_progress_bar=False)

    clusters = kmeans.predict(embeddings)
    df["topic_cluster"] = clusters

    return df
