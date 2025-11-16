import os
import pytest
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# 🔹 Charge automatiquement le fichier .env à la racine du projet
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")


@pytest.mark.skipif(
    not DATABASE_URL,
    reason="DATABASE_URL non défini — test ignoré en environnement local."
)
def test_neondb_connection():
    """Vérifie la connexion à NeonDB."""
    engine = create_engine(DATABASE_URL)
    with engine.connect() as conn:
        result = conn.execute(text("SELECT 1;"))
        assert result.scalar() == 1, "❌ Connexion NeonDB échouée"
    print("✅ Connexion NeonDB réussie.")


@pytest.mark.skipif(
    not DATABASE_URL,
    reason="DATABASE_URL non défini — test ignoré en environnement local."
)
def test_reddit_scoring_not_empty():
    """Vérifie que la table reddit_scoring contient des données."""
    engine = create_engine(DATABASE_URL)
    query = "SELECT COUNT(*) AS n FROM reddit_scoring;"
    df = pd.read_sql(query, engine)

    assert not df.empty, "❌ Requête invalide, table introuvable."
    assert df.loc[0, "n"] > 0, "❌ Table reddit_scoring vide."
    print(f"✅ Table reddit_scoring contient {df.loc[0, 'n']} lignes.")


@pytest.mark.skipif(
    not DATABASE_URL,
    reason="DATABASE_URL non défini — test ignoré en environnement local."
)
def test_emotion_columns_exist():
    """Vérifie que les colonnes d'émotions sont présentes dans reddit_scoring."""
    engine = create_engine(DATABASE_URL)
    query = """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_name = 'reddit_scoring';
    """
    cols = pd.read_sql(query, engine)["column_name"].tolist()

    expected = {"joy", "anger", "sadness", "surprise", "disgust", "fear", "neutral"}
    missing = expected - set(cols)

    assert not missing, f"❌ Colonnes émotion manquantes : {missing}"
    print("✅ Toutes les colonnes émotionnelles sont présentes :", expected)
