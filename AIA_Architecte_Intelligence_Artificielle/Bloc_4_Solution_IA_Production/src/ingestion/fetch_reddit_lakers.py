import requests
import pandas as pd
from datetime import datetime
from pathlib import Path
import time

# === CONFIGURATION ===
SUBREDDIT = "lakers"
LIMIT = 100

# Déterminer la racine du projet (2 niveaux au-dessus du script)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def fetch_reddit_posts(subreddit=SUBREDDIT, limit=LIMIT):
    """Récupère les derniers posts du subreddit Reddit (API publique)."""
    url = f"https://www.reddit.com/r/{subreddit}/new.json?limit={limit}"
    headers = {"User-Agent": "nba-mood-monitor/0.1"}
    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        raise Exception(f"Erreur {response.status_code} : {response.text}")

    data = response.json().get("data", {}).get("children", [])
    print(f"✅ {len(data)} posts récupérés depuis r/{subreddit}")
    return data


def parse_posts(raw_posts):
    """Structure les données Reddit dans un DataFrame propre."""
    posts = []
    for p in raw_posts:
        d = p.get("data", {})
        posts.append({
            "id": d.get("id"),
            "created_utc": datetime.utcfromtimestamp(d.get("created_utc", 0)).isoformat(),
            "title": d.get("title"),
            "selftext": d.get("selftext"),
            "author": d.get("author"),
            "score": d.get("score"),
            "upvote_ratio": d.get("upvote_ratio"),
            "num_comments": d.get("num_comments"),
            "over_18": d.get("over_18"),
            "stickied": d.get("stickied"),
            "permalink": f"https://reddit.com{d.get('permalink')}",
            "url": d.get("url"),
            "subreddit": d.get("subreddit"),
            "flair": d.get("link_flair_text"),
            "is_video": d.get("is_video"),
            "domain": d.get("domain"),
            "thumbnail": d.get("thumbnail"),
        })
    return pd.DataFrame(posts)


def save_file(df):
    """Sauvegarde les données dans le dossier data/raw sous format CSV."""
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename = OUTPUT_DIR / f"reddit_{SUBREDDIT}_{timestamp}.csv"
    df.to_csv(filename, index=False, encoding="utf-8")
    print(f"💾 Données sauvegardées dans {filename}")
    return filename


if __name__ == "__main__":
    try:
        print("🔄 Lancement de la collecte Reddit...")
        raw_data = fetch_reddit_posts()
        df = parse_posts(raw_data)
        save_file(df)
        print(f"✅ Fin du script : {len(df)} posts enregistrés avec succès.")
    except Exception as e:
        print("❌ Erreur :", e)
        time.sleep(3)
