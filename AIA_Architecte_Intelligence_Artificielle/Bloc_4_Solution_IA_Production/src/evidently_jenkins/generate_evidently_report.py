import os
import glob
from pathlib import Path
from datetime import datetime

import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataQualityPreset


# Base du projet : .../Bloc_4_Solution_IA_Production
BASE_DIR = Path(__file__).resolve().parents[2]


def get_latest_scored_file() -> Path:
    """Retourne le dernier fichier de scoring dans data/scored/."""
    scored_dir = BASE_DIR / "data" / "scored"
    pattern = scored_dir / "reddit_scoring_*.csv"
    files = glob.glob(str(pattern))

    if not files:
        raise FileNotFoundError(f"Aucun fichier de scoring trouvé dans {scored_dir}")

    latest = max(files, key=os.path.getmtime)
    return Path(latest)


def main():
    # 1️⃣ Charger le dernier fichier scoré
    latest_file = get_latest_scored_file()
    print(f"[Evidently] Utilisation du fichier de scoring : {latest_file}")
    df = pd.read_csv(latest_file)

    # 2️⃣ Construire le rapport Evidently
    report = Report(metrics=[DataQualityPreset()])

    # ⚠️ IMPORTANT : Evidently 0.4.20 → utiliser des arguments nommés
    report.run(current_data=df)

    # 3️⃣ Sauvegarde du rapport HTML dans data/reports/jenkins/
    reports_dir = BASE_DIR / "data" / "reports" / "jenkins"
    reports_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M")
    output_path = reports_dir / f"reddit_data_quality_jenkins_{ts}.html"

    report.save_html(str(output_path))
    print(f"[Evidently] Rapport sauvegardé dans : {output_path}")


if __name__ == "__main__":
    main()
