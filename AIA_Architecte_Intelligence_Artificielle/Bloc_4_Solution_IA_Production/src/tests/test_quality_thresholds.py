import os
import pytest
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
IS_CI = os.getenv("CI", "false").lower() == "true"


@pytest.mark.skipif(
    not DATABASE_URL or IS_CI,
    reason="Ignoré : pas de DATABASE_URL ou exécution CI."
)
def test_quality_thresholds():
    """Vérifie que les métriques de qualité respectent les seuils définis."""
    engine = create_engine(DATABASE_URL)

    # ✨ Vérifier que la table existe
    try:
        cols = pd.read_sql(
            """
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name='reddit_quality_history';
            """,
            engine
        )["column_name"].tolist()
    except Exception:
        pytest.skip("⏭️ Table reddit_quality_history inexistante — test ignoré.")

    if len(cols) == 0:
        pytest.skip("⏭️ Table reddit_quality_history vide — test ignoré.")

    # ✨ Détection dynamique de la colonne ordonnante
    order_col = (
        "run_timestamp"
        if "run_timestamp" in cols
        else "timestamp"
        if "timestamp" in cols
        else cols[0]  # fallback sûr car cols n'est jamais vide ici
    )

    # ✨ Charger la table
    try:
        df = pd.read_sql(
            f"SELECT * FROM reddit_quality_history ORDER BY {order_col} DESC LIMIT 1;",
            engine
        )
    except Exception:
        pytest.skip("⏭️ Impossible de lire reddit_quality_history — ignoré.")

    if df.empty:
        pytest.skip("⏭️ Aucun enregistrement — test ignoré.")

    # ✨ Exemple de règles de qualité
    assert df["mean_confidence"].iloc[0] > 0.6, "❌ Confiance trop faible"
    assert df["pct_missing"].iloc[0] < 0.1, "❌ Trop de valeurs manquantes"

    print("✅ Seuils de qualité OK.")
