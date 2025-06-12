import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

# 📁 Chemins
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
IMG_DIR = os.path.join(BASE_DIR, "src_pictures", "test_skindisease")
MODEL_PATH = os.path.join(BASE_DIR, "models", "4_skindisease_efficientnetb0_finetuned", "model4_best_model.h5")

# 📦 Classes
CLASS_NAMES = ['benign', 'malignant']

# 🔍 Choix aléatoire d’une image
def pick_random_image(img_dir):
    class_folder = random.choice(CLASS_NAMES)
    folder_path = os.path.join(img_dir, class_folder)
    image_file = random.choice(os.listdir(folder_path))
    return os.path.join(folder_path, image_file), class_folder

# 🧠 Prédiction
def predict_image(model, img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_preprocessed = preprocess_input(np.expand_dims(img_array, axis=0))

    # Tqdm pour "effet de chargement"
    for _ in tqdm(range(10), desc="Prédiction en cours", ncols=80):
        time.sleep(0.05)  # Simulation courte

    prob = model.predict(img_preprocessed, verbose=0)[0][0]
    predicted_class = 'malignant' if prob >= 0.5 else 'benign'
    prob_malignant = prob * 100
    prob_benign = 100 - prob_malignant

    return predicted_class, prob_benign, prob_malignant, img

# ▶️ Main
if __name__ == "__main__":
    print("🔧 Chargement du modèle...")
    model = load_model(MODEL_PATH)

    print("🖼️  Sélection aléatoire d’une image...")
    img_path, true_class = pick_random_image(IMG_DIR)

    pred_class, prob_benign, prob_malignant, img = predict_image(model, img_path)

    print(f"\n📷 Image sélectionnée : {img_path}")
    print(f"✅ Classe réelle       : {true_class}")
    print(f"🤖 Classe prédite      : {pred_class}")
    print(f"📊 Probabilités        : benign = {prob_benign:.2f}% | malignant = {prob_malignant:.2f}%")

    # 🖼️ Affichage de l'image avec titre
    plt.imshow(np.array(img))
    plt.title(f"Vérité : {true_class} | Prédit : {pred_class}")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

