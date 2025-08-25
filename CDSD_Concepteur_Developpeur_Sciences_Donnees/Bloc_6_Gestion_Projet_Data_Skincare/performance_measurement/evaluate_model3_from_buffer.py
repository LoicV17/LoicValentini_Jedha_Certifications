import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# -----------------------------------------------------------------------------
# Config chemins
# -----------------------------------------------------------------------------
CUR_DIR = os.path.abspath(os.path.dirname(__file__))
BASE_DIR = os.path.abspath(os.path.join(CUR_DIR, ".."))

LOCAL_STREAMLIT_DIR = os.path.join(BASE_DIR, "local_streamlit")
SRC_PICTURES_DIR = os.path.join(BASE_DIR, "src_pictures")

BUFFER_CSV = os.path.join(CUR_DIR, "m2_probs_buffer.csv")
METADATA_CSV = os.path.join(SRC_PICTURES_DIR, "initial", "HAM10000_metadata.csv")
MODEL3_PATH = os.path.join(LOCAL_STREAMLIT_DIR, "model3.joblib")

OUTPUT_DIR = CUR_DIR
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# Classes
# -----------------------------------------------------------------------------
CLASS_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

# -----------------------------------------------------------------------------
# Chargement buffer, métadonnées, modèle
# -----------------------------------------------------------------------------
print("🔁 Chargement buffer, métadonnées et modèle...")
m2_df = pd.read_csv(BUFFER_CSV)
meta = pd.read_csv(METADATA_CSV)
stacking_model = joblib.load(MODEL3_PATH)

# Jointure métadonnées
if "image_id" not in meta.columns:
    raise KeyError(f"'image_id' introuvable dans {METADATA_CSV}. Colonnes: {list(meta.columns)}")

merged = m2_df.merge(meta[["image_id", "age", "sex", "localization"]], on="image_id", how="left")

# Renommer colonnes proba
rename_map = {f"m2_prob_{cls}": f"proba_{cls}" for cls in CLASS_NAMES}
merged = merged.rename(columns=rename_map)

# Features d’entrée modèle 3
proba_cols = [f"proba_{cls}" for cls in CLASS_NAMES]
feature_cols = proba_cols + ["age", "sex", "localization"]

# -----------------------------------------------------------------------------
# Prédiction et métriques
# -----------------------------------------------------------------------------
print("🚀 Prédictions modèle 3...")
y_true = merged["true_label"]
y_pred = stacking_model.predict(merged[feature_cols])

print("\n✅ Rapport de classification - Modèle 3 (Stacking RF) :")
report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=3)
print(report)

with open(os.path.join(OUTPUT_DIR, "classification_report_model3.txt"), "w") as f:
    f.write(report)

# -----------------------------------------------------------------------------
# Matrice de confusion
# -----------------------------------------------------------------------------
cm = confusion_matrix(y_true, y_pred, labels=CLASS_NAMES, normalize="true") * 100

plt.figure(figsize=(10, 7))
sns.heatmap(
    cm, annot=True, fmt=".1f", cmap="Blues",
    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES
)
plt.title("Matrice de confusion (%) - Modèle 3 (Stacking)")
plt.xlabel("Prédit")
plt.ylabel("Réel")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix_model3.png"))
plt.close()

print(f"\n💾 Rapport et matrice de confusion sauvegardés dans : {OUTPUT_DIR}")
