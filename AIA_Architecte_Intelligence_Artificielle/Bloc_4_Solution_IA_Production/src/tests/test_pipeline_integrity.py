import os
import pytest
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# 🔹 Charger les variables d'environnement
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
IS_CI = os.getenv("CI", "false").lower() == "true"


# ============================================================
#  TEST 1 — Connexion à NeonDB
# ============================================================
@pytest.mark.skipif(
    not DATABASE_URL or IS_CI,
    reason="Ignoré : pas de DATABASE_URL ou exécution CI."
)
def test_neondb_connection():
    """Vérifie la connexion à NeonDB."""
    engine = create_engine(DATABASE_URL)
    with engine.connect() as conn:
        result = conn.execute(text("SELECT 1;"))
        assert result.scalar() == 1, "❌ Connexion NeonDB échouée"
    print("✅ Connexion NeonDB réussie.")


# ============================================================
#  TEST 2 — Table reddit_scoring non vide
# ============================================================
@pytest.mark.skipif(
    not DATABASE_URL or IS_CI,
    reason="Ignoré : pas de DATABASE_URL ou exécution CI."
)
def test_reddit_scoring_not_empty():
    """Vérifie que la table reddit_scoring contient des données."""
    engine = create_engine(DATABASE_URL)
    query = "SELECT COUNT(*) AS n FROM reddit_scoring;"

    try:
        df = pd.read_sql(query, engine)
    except Exception:
        pytest.skip("⏭️ Table reddit_scoring absente — ignoré.")

    assert not df.empty, "❌ Requête invalide."
    assert df.loc[0, "n"] > 0, "❌ Table reddit_scoring vide."

    print(f"✅ Table reddit_scoring contient {df.loc[0, 'n']} lignes.")


# ============================================================
#  TEST 3 — Colonnes émotionnelles présentes
# ============================================================
@pytest.mark.skipif(
    not DATABASE_URL or IS_CI,
    reason="Ignoré : pas de DATABASE_URL ou exécution CI."
)
def test_emotion_columns_exist():
    """Vérifie que les colonnes d'émotions existent dans reddit_scoring."""
    engine = create_engine(DATABASE_URL)

    query = """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_name = 'reddit_scoring';
    """

    try:
        cols = pd.read_sql(query, engine)["column_name"].tolist()
    except Exception:
        pytest.skip("⏭️ Table reddit_scoring absente — ignoré.")

    expected = {"joy", "anger", "sadness", "surprise", "disgust", "fear", "neutral"}
    missing = expected - set(cols)

    assert not missing, f"❌ Colonnes émotion manquantes : {missing}"

    print("✅ Toutes les colonnes émotionnelles sont présentes :", expected)
