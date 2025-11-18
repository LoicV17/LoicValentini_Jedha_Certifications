import os
import glob
from pathlib import Path
from datetime import datetime

import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataQualityPreset


BASE_DIR = Path(__file__).resolve().parents[2]


def get_scored_files():
    """Retourne tous les fichiers scorés triés par date."""
    directory = BASE_DIR / "data" / "scored"
    files = sorted(
        glob.glob(str(directory / "reddit_scoring_*.csv")),
        key=os.path.getmtime
    )
    return files


def main():
    files = get_scored_files()

    if len(files) == 0:
        raise FileNotFoundError("Aucun fichier trouvé dans data/scored/")

    # 1️⃣ CURRENT DATA = dernier fichier
    current_path = Path(files[-1])
    df_current = pd.read_csv(current_path)
    print(f"[Evidently] Fichier actuel : {current_path}")

    # 2️⃣ REFERENCE DATA
    if len(files) >= 2:
        # Baseline = fichier précédant
        reference_path = Path(files[-2])
        df_reference = pd.read_csv(reference_path)
        print(f"[Evidently] Référence : {reference_path}")
    else:
        # fallback minimal → 20 premières lignes du fichier actuel
        df_reference = df_current.head(20)
        print("[Evidently] Référence minimale générée (pas de fichier précédent)")

    # 3️⃣ Construire le rapport Evidently
    report = Report(metrics=[DataQualityPreset()])

    # IMPORTANT : Evidently 0.4.20 impose reference_data=...
    report.run(
        reference_data=df_reference,
        current_data=df_current
    )

    # 4️⃣ Sauvegarde
    reports_dir = BASE_DIR / "data" / "reports" / "jenkins"
    reports_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M")
    output_path = reports_dir / f"jenkins_reddit_data_quality_{timestamp}.html"

    report.save_html(str(output_path))
    print(f"[Evidently] Rapport généré : {output_path}")


if __name__ == "__main__":
    main()
