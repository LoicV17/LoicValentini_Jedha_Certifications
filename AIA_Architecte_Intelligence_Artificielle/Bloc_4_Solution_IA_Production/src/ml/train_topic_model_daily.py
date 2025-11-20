import os
import json
import shutil
from datetime import datetime
from urllib.parse import urlparse, parse_qs

import pandas as pd
import numpy as np
import psycopg2
import joblib

from sqlalchemy import create_engine
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sentence_transformers import SentenceTransformer

# ───────────────────────────────────────────────────────
# ENV
# ───────────────────────────────────────────────────────
DB_URL = os.getenv("DATABASE_URL")
if not DB_URL:
    raise ValueError("❌ DB_URL est introuvable dans les variables d'environnement.")

engine = create_engine(DB_URL)

MODEL_DIR = "/opt/airflow/src/models/topic_clustering"
os.makedirs(MODEL_DIR, exist_ok=True)

# ───────────────────────────────────────────────────────
# Convertisseur SQLAlchemy → psycopg2
# ───────────────────────────────────────────────────────
def sqlalchemy_url_to_psycopg2_dsn(url):
    parsed = urlparse(url)

    dbname = parsed.path.lstrip("/")
    user = parsed.username
    password = parsed.password
    host = parsed.hostname
    port = parsed.port or 5432

    query = parse_qs(parsed.query)
    sslmode = query.get("sslmode", ["require"])[0]

    return (
        f"dbname={dbname} user={user} password={password} "
        f"host={host} port={port} sslmode={sslmode}"
    )

# ───────────────────────────────────────────────────────
# 1 — Chargement des posts
# ───────────────────────────────────────────────────────
def load_posts():
    query = """
    SELECT
        id,
        title,
        selftext,
        COALESCE(title, '') || ' ' || COALESCE(selftext, '') AS text
    FROM reddit_cleaned
    WHERE title IS NOT NULL OR selftext IS NOT NULL
    """

    df = pd.read_sql(query, engine)
    df.dropna(subset=["text"], inplace=True)
    df["text"] = df["text"].astype(str)
    return df

# ───────────────────────────────────────────────────────
# 2 — Embeddings
# ───────────────────────────────────────────────────────
def embed_posts(texts):
    model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L6-v2")
    return model.encode(texts, batch_size=64, show_progress_bar=True)

# ───────────────────────────────────────────────────────
# 3 — Top keywords TF-IDF par cluster
# ───────────────────────────────────────────────────────
def extract_cluster_keywords(df, vectors, n_clusters, vectorizer, top_k=12, column="cluster"):
    keywords = {}
    feature_names = vectorizer.get_feature_names_out()

    for cluster_id in range(n_clusters):
        mask = (df[column] == cluster_id).values
        cluster_docs = vectors[mask]

        if cluster_docs.shape[0] == 0:
            keywords[cluster_id] = []
            continue

        mean_tfidf = cluster_docs.mean(axis=0)
        mean_tfidf = np.array(mean_tfidf).ravel()

        top_idx = mean_tfidf.argsort()[::-1][:top_k]
        keywords[cluster_id] = feature_names[top_idx].tolist()

    return keywords

# ───────────────────────────────────────────────────────
# 4 — Calcul métriques clustering
# ───────────────────────────────────────────────────────
def compute_metrics(df, vectors, kmeans):
    labels = df["cluster"]

    inertia = kmeans.inertia_
    silhouette = silhouette_score(vectors, labels) if len(set(labels)) > 1 else -1.0

    cluster_sizes = df["cluster"].value_counts().sort_index().to_dict()

    n_outliers = int((df["cluster"] == -1).sum()) if -1 in cluster_sizes else 0

    return {
        "n_samples": len(df),
        "n_clusters": len(cluster_sizes),
        "inertia": float(inertia),
        "silhouette": float(silhouette),
        "cluster_sizes": cluster_sizes,
        "n_outliers": n_outliers,
    }

# ───────────────────────────────────────────────────────
# 5 — Sauvegarde modèle + métriques
# ───────────────────────────────────────────────────────
def save_model_and_metrics(kmeans, metrics, cluster_keywords):

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    versioned_path = f"{MODEL_DIR}/cluster_model_{timestamp}.joblib"
    latest_path = f"{MODEL_DIR}/latest.joblib"

    joblib.dump(kmeans, versioned_path)

    if os.path.exists(latest_path):
        os.remove(latest_path)
    shutil.copy2(versioned_path, latest_path)

    # Sauvegarde SQL
    dsn = sqlalchemy_url_to_psycopg2_dsn(DB_URL)
    conn = psycopg2.connect(dsn)
    cur = conn.cursor()

    metrics_json_keywords = json.dumps(cluster_keywords)
    metrics_json_sizes = json.dumps(metrics["cluster_sizes"])

    query = """
        INSERT INTO reddit_topic_models_metrics (
            run_id,
            model_path,
            n_samples,
            n_clusters,
            inertia,
            silhouette,
            n_outliers,
            cluster_sizes,
            cluster_keywords
        )
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s::jsonb)
    """

    # run_id = horodatage unique ISO
    run_id = datetime.utcnow().isoformat()

    cur.execute(
        query,
        (
            run_id,
            versioned_path,
            metrics["n_samples"],
            metrics["n_clusters"],
            metrics["inertia"],
            metrics["silhouette"],
            metrics["n_outliers"],
            metrics_json_sizes,
            metrics_json_keywords,
        ),
    )

    conn.commit()
    cur.close()
    conn.close()

# ───────────────────────────────────────────────────────
# MAIN PIPELINE
# ───────────────────────────────────────────────────────
def main():

    n_clusters = 8

    print("\n────────────────────────────────────────────────────")
    print("🚀 Début du réentraînement quotidien du modèle")
    print("────────────────────────────────────────────────────\n")

    # 1) Chargement des posts
    df = load_posts()
    print(f"📌 {len(df)} posts chargés depuis la base.")

    if df.empty:
        raise ValueError("❌ Aucun post disponible pour l'entraînement.")

    # 2) TF-IDF
    print("🔧 Vectorisation TF-IDF...")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words="english"
    )
    tfidf_vectors = vectorizer.fit_transform(df["text"])
    print("✅ TF-IDF terminé.")

    # 3) KMeans
    print("🔧 Clustering KMeans...")
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init="auto"
    )
    df["cluster"] = kmeans.fit_predict(tfidf_vectors)
    print(f"✅ Clustering terminé : {n_clusters} clusters générés.")

    # 4) Métriques
    print("📊 Calcul des métriques...")
    metrics = compute_metrics(df, tfidf_vectors, kmeans)
    print(f"   → silhouette_score = {metrics['silhouette']:.4f}")
    print(f"   → inertie = {metrics['inertia']:.2f}")

    # 5) Keywords
    print("🧠 Extraction des mots-clés...")
    cluster_keywords = extract_cluster_keywords(
        df,
        tfidf_vectors,
        n_clusters,
        vectorizer
    )
    print("✅ Extraction terminée.")

    # 6) Sauvegardes
    print("💾 Sauvegarde du modèle et des métriques...")
    save_model_and_metrics(kmeans, metrics, cluster_keywords)
    print("✅ Sauvegarde réussie.")

    print("\n────────────────────────────────────────────────────")
    print("🎉 Réentraînement terminé avec succès !")
    print("────────────────────────────────────────────────────\n")

# ───────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
