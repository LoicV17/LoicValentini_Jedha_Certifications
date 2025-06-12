import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
from sklearn.metrics import accuracy_score, recall_score, f1_score
from tqdm import tqdm

# 📁 Chemins
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
IMG_DIR = os.path.join(BASE_DIR, "src_pictures", "test_skindisease")
MODEL_PATH = os.path.join(BASE_DIR, "models", "4_skindisease_efficientnetb0_finetuned", "model4_best_model.h5")

# 📦 Classes
CLASS_NAMES = ['benign', 'malignant']
CLASS_TO_INT = {'benign': 0, 'malignant': 1}

# 🧠 Chargement du modèle
model = load_model(MODEL_PATH)

# 📦 Collecte des fichiers
image_paths = []
true_labels = []

for label in CLASS_NAMES:
    folder = os.path.join(IMG_DIR, label)
    for img_file in os.listdir(folder):
        image_paths.append(os.path.join(folder, img_file))
        true_labels.append(CLASS_TO_INT[label])

# 🔍 Prédictions avec barre de progression
results = []

for img_path, true_label in tqdm(zip(image_paths, true_labels), total=len(image_paths), desc="Prédictions", ncols=80):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_preprocessed = preprocess_input(np.expand_dims(img_array, axis=0))

    prob = model.predict(img_preprocessed, verbose=0)[0][0]
    pred_label = 1 if prob >= 0.5 else 0
    error = int(pred_label != true_label)

    results.append({
        "image": os.path.basename(img_path),
        "true_class": CLASS_NAMES[true_label],
        "pred_class": CLASS_NAMES[pred_label],
        "error": error,
        "proba_benign": round(100 - prob * 100, 2),
        "proba_malignant": round(prob * 100, 2)
    })

# 📊 DataFrame
df = pd.DataFrame(results)

# 📈 Scores globaux
accuracy = accuracy_score(df["true_class"], df["pred_class"])
recall = recall_score(df["true_class"], df["pred_class"], pos_label="malignant")
f1 = f1_score(df["true_class"], df["pred_class"], pos_label="malignant")

# 🔎 Résumé
print("\n✅ Résultats globaux sur toutes les images test :")
print(f"Accuracy : {accuracy:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")

# 🔎 Aperçu du DataFrame
print("\n📝 Aperçu des 5 premières lignes du DataFrame :")
print(df.head())

# 💾 Sauvegarde en CSV
output_csv_path = os.path.join(BASE_DIR, "test_model", "test_predictions.csv")
df.to_csv(output_csv_path, index=False)
print(f"\n✅ Résultats enregistrés dans : {output_csv_path}")

