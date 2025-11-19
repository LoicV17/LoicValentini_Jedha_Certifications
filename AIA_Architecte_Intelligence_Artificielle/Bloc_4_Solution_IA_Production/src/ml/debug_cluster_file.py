import joblib
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "topic_clustering" / "cluster_model.joblib"

print("➡️ Chargement du fichier:", MODEL_PATH)
obj = joblib.load(MODEL_PATH)

print("➡️ Type :", type(obj))
print("➡️ Contenu :", obj)
