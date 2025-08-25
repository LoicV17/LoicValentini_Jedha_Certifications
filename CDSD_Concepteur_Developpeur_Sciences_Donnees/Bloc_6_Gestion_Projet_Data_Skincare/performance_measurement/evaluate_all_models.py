import os
import sys
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input as effnet_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess

import joblib

# -----------------------------------------------------------------------------
# Configuration chemins (structure projet)
# -----------------------------------------------------------------------------
# Ce script est placé dans: Bloc_6_Gestion_Projet_Data_Skincare/performance_measurement/
CUR_DIR = os.path.abspath(os.path.dirname(__file__))
BASE_DIR = os.path.abspath(os.path.join(CUR_DIR, ".."))  # Bloc_6_Gestion_Projet_Data_Skincare

LOCAL_STREAMLIT_DIR = os.path.join(BASE_DIR, "local_streamlit")
SRC_PICTURES_DIR = os.path.join(BASE_DIR, "src_pictures")
TEST_DIR = os.path.join(SRC_PICTURES_DIR, "test_ham10000")
METADATA_CSV = os.path.join(SRC_PICTURES_DIR, "initial", "HAM10000_metadata.csv")

OUTPUT_DIR = CUR_DIR  # performance_measurement
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Modèles
MODEL1_PATH = os.path.join(LOCAL_STREAMLIT_DIR, "model1_h5version.h5")  # binaire
MODEL2_PATH = os.path.join(LOCAL_STREAMLIT_DIR, "model2.h5")            # 7 classes
MODEL3_PATH = os.path.join(LOCAL_STREAMLIT_DIR, "model3.joblib")        # stacking RF

# Buffer CSV (probas modèle 2) + sortie finale
M2_BUFFER_CSV = os.path.join(OUTPUT_DIR, "m2_probs_buffer.csv")
SUMMARY_CSV   = os.path.join(OUTPUT_DIR, "performance_summary.csv")

# -----------------------------------------------------------------------------
# Classes et mapping
# -----------------------------------------------------------------------------
# Ordre ALPHABÉTIQUE confirmé
CLASS_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
MALIGNANT = {'akiec', 'bcc', 'mel'}
BENIGN   = {'bkl', 'df', 'nv', 'vasc'}

# -----------------------------------------------------------------------------
# Chargement des modèles
# -----------------------------------------------------------------------------
print("🔁 Chargement des modèles...")
model1 = load_model(MODEL1_PATH)  # binaire, sortie sigmoid (proba malignant)
model2 = load_model(MODEL2_PATH)  # multiclasses (7)
stacking_model = joblib.load(MODEL3_PATH)  # RandomForest pipeline (preproc inside)

print("✅ Modèles chargés.")

# -----------------------------------------------------------------------------
# Utilitaires image
# -----------------------------------------------------------------------------
def load_and_preprocess_for_model1(img_path):
    """240x240 EfficientNet preprocess"""
    img = image.load_img(img_path, target_size=(240, 240))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = effnet_preprocess(x)
    return x

def load_and_preprocess_for_model2(img_path):
    """224x224 ResNet50 preprocess"""
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = resnet_preprocess(x)
    return x

def image_id_from_path(p):
    """Extrait ISIC_XXXXXX à partir du nom de fichier (sans extension)."""
    return os.path.splitext(os.path.basename(p))[0]

# -----------------------------------------------------------------------------
# Collecte des chemins d'images test
# -----------------------------------------------------------------------------
image_records = []  # (img_path, true_label)
for cls in CLASS_NAMES:
    cls_dir = os.path.join(TEST_DIR, cls)
    if not os.path.isdir(cls_dir):
        print(f"⚠️  Dossier classe manquant (ignoré) : {cls_dir}")
        continue
    for img_path in glob.glob(os.path.join(cls_dir, "*.jpg")):
        image_records.append((img_path, cls))

if not image_records:
    print("❌ Aucune image trouvée dans test_ham10000/. Vérifie les chemins.")
    sys.exit(1)

print(f"🖼️ Images détectées: {len(image_records)}")

# -----------------------------------------------------------------------------
# Passage dans les modèles 1 & 2 (probas)
# -----------------------------------------------------------------------------
if os.path.exists(M2_BUFFER_CSV):
    print(f"🔁 Utilisation du tampon existant : {M2_BUFFER_CSV}")
    m2_df = pd.read_csv(M2_BUFFER_CSV)
else:
    rows = []
    print("🚀 Inférence modèles 1 & 2...")
    for img_path, true_label in tqdm(image_records, ncols=80):
        img_id = image_id_from_path(img_path)

        # Modèle 1 (binaire) — sortie sigmoid interprétée comme p(malignant)
        x1 = load_and_preprocess_for_model1(img_path)
        p_mal = float(model1.predict(x1, verbose=0).ravel()[0])
        p_ben = 1.0 - p_mal

        # Modèle 2 (7 classes)
        x2 = load_and_preprocess_for_model2(img_path)
        probs = model2.predict(x2, verbose=0).ravel()
        if probs.shape[0] != len(CLASS_NAMES):
            raise ValueError(
                f"Les dimensions de sortie du modèle 2 ({probs.shape[0]}) "
                f"ne correspondent pas au nombre de classes ({len(CLASS_NAMES)})."
            )

        # Probas par classe
        m2_dict = {f"m2_prob_{cls}": float(probs[i]) for i, cls in enumerate(CLASS_NAMES)}

        # Sommes bénin/malin
        m2_prob_malignant_sum = float(sum(m2_dict[f"m2_prob_{c}"] for c in MALIGNANT))
        m2_prob_benign_sum = float(sum(m2_dict[f"m2_prob_{c}"] for c in BENIGN))

        row = {
            "image_id": img_id,
            "true_label": true_label,
            # Model 1
            "m1_prob_benign": p_ben,
            "m1_prob_malignant": p_mal,
            # Model 2 (7 classes)
            **m2_dict,
            "m2_prob_benign_sum": m2_prob_benign_sum,
            "m2_prob_malignant_sum": m2_prob_malignant_sum,
        }
        rows.append(row)

    m2_df = pd.DataFrame(rows)
    m2_df.to_csv(M2_BUFFER_CSV, index=False)
    print(f"💾 Tampon modèle 2 sauvegardé: {M2_BUFFER_CSV}")


# -----------------------------------------------------------------------------
# Jointure métadonnées et prédiction modèle 3 (stacking)
# -----------------------------------------------------------------------------
print("🔗 Jointure métadonnées & prédictions stacking...")

# Chargement métadonnées
meta = pd.read_csv(METADATA_CSV)

# Vérifie/adapter le nom de colonne ID
if "image_id" not in meta.columns:
    # Si jamais ton CSV a une autre colonne (ex. 'image' sans extension), adapte ici.
    raise KeyError(f"'image_id' introuvable dans {METADATA_CSV}. Colonnes disponibles: {list(meta.columns)}")

# Merge
merged = m2_df.merge(meta[["image_id", "age", "sex", "localization"]], on="image_id", how="left")

# Renommer m2_prob_* -> proba_* pour coller aux attentes du pipeline joblib
rename_map = {f"m2_prob_{cls}": f"proba_{cls}" for cls in CLASS_NAMES}
merged = merged.rename(columns=rename_map)

# Colonnes attendues par le pipeline du stacking
proba_cols = [f"proba_{cls}" for cls in CLASS_NAMES]
feature_cols = proba_cols + ["age", "sex", "localization"]

# (Optionnel) s’assurer que les proba_* sont bien float
merged[proba_cols] = merged[proba_cols].astype(float)

# Prédiction stacking (le pipeline gère imputation âge + OneHot sex/localization)
m3_preds = stacking_model.predict(merged[feature_cols])
merged["m3_pred_class"] = m3_preds

# Sauvegarde finale
merged.to_csv(SUMMARY_CSV, index=False)
print(f"✅ Fichier récapitulatif sauvegardé : {SUMMARY_CSV}")

