import pandas as pd
from pathlib import Path
import re
from datetime import datetime
import argparse
from typing import Optional

# === CONFIG ===
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def clean_text(text: str) -> str:
    if pd.isna(text):
        return ""
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s.,!?']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def clean_incremental(latest_file: Optional[Path] = None):
    # 1) Choisir le fichier brut à traiter
    if latest_file is None:
        raw_files = sorted(RAW_DIR.glob("reddit_lakers_*.csv"),
                           key=lambda f: f.stat().st_mtime, reverse=True)
        if not raw_files:
            print("⚠️ Aucun fichier brut trouvé.")
            return
        latest_file = raw_files[0]
    else:
        latest_file = Path(latest_file)

    print(f"🆕 Fichier brut: {latest_file.name}")
    new_df = pd.read_csv(latest_file)
    print(f"📥 {len(new_df)} lignes importées.")

    # 2) Nettoyage texte
    for col in ("title", "selftext"):
        if col in new_df.columns:
            new_df[col] = new_df[col].astype(str).apply(clean_text)

    # 3) Filtre simples: lignes “non vides”
    new_df = new_df[(new_df["title"].str.len() > 5) | (new_df["selftext"].str.len() > 10)]

    # 4) Fusion avec dernier dataset nettoyé (incrémental + dédoublonnage global)
    processed_files = sorted(PROCESSED_DIR.glob("reddit_cleaned_*.csv"),
                             key=lambda f: f.stat().st_mtime, reverse=True)
    if processed_files:
        latest_cleaned = processed_files[0]
        df_old = pd.read_csv(latest_cleaned)
        print(f"🔁 Fusion avec {latest_cleaned.name} ({len(df_old)} lignes).")
        df_final = pd.concat([df_old, new_df], ignore_index=True)
        df_final.drop_duplicates(subset=["id"], inplace=True)
    else:
        print("⚙️ Aucun dataset précédent — création du premier consolidé.")
        df_final = new_df

    print(f"📊 {len(df_final)} lignes après dédoublonnage.")

    # 5) Sauvegarde versionnée
    out = PROCESSED_DIR / f"reddit_cleaned_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv"
    df_final.to_csv(out, index=False, encoding="utf-8")
    print(f"✅ Nettoyage sauvegardé : {out}")
    return str(out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--latest-file", type=str, default=None,
                        help="Chemin du dernier CSV brut à traiter")
    args = parser.parse_args()
    clean_incremental(args.latest_file)
