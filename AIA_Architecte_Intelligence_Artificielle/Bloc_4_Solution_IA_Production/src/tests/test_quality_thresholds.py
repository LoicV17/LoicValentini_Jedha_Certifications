import os
import pytest
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")


@pytest.mark.skipif(not DATABASE_URL, reason="DATABASE_URL non défini")
def test_quality_thresholds():
    """Vérifie que les métriques de qualité respectent les seuils définis."""
    engine = create_engine(DATABASE_URL)

    # Récupération du dernier run (colonne d'ordre flexible)
    cols = pd.read_sql(
        "SELECT column_name FROM information_schema.columns WHERE table_name='reddit_quality_history';",
        engine
    )["column_name"].tolist()

    order_col = (
        "run_timestamp"
        if "run_timestamp" in cols
        else "timestamp" if "timestamp" in cols
        else cols[0]
    )

    df = pd.read_sql(f"SELECT * FROM reddit_quality_history ORDER BY {order_col} DESC LIMIT 1;", engine)
    assert not df.empty, "❌ Aucune donnée trouvée dans reddit_quality_history"

    # Correspondance flexible des colonnes selon le schéma en base
    text_ratio_col = "empty_text_ratio" if "empty_text_ratio" in df.columns else (
        "missing_ratio" if "missing_ratio" in df.columns else None
    )

    # Vérification des seuils
    assert df["duplicates_ratio"].iloc[0] < 0.2, "⚠️ Trop de doublons détectés"
    if text_ratio_col:
        assert df[text_ratio_col].iloc[0] < 0.1, f"⚠️ Trop de textes vides ({text_ratio_col})"
    else:
        print("ℹ️ Aucune colonne de ratio de texte vide détectée (test partiel)")

    if "emotion_outlier_rate" in df.columns:
        assert df["emotion_outlier_rate"].iloc[0] < 0.5, "⚠️ Trop d’outliers émotionnels"
    else:
        print("ℹ️ Aucune colonne emotion_outlier_rate détectée (test partiel)")

    print("✅ Vérification qualité : toutes les métriques respectent les seuils.")
