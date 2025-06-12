import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing import image
from tqdm import tqdm

# 📁 Paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
IMG_DIR = os.path.join(BASE_DIR, "src_pictures")
TRAIN_DIR = os.path.join(IMG_DIR, "train2_ham10000")
VAL_DIR = os.path.join(IMG_DIR, "val2_ham10000")
METADATA_PATH = os.path.join(IMG_DIR, "initial", "HAM10000_metadata.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "7_ham10000_efficientnetb0_finetuned", "model7_bestmodel.h5")
STACKING_DIR = os.path.join(BASE_DIR, "models", "stacking_model")
os.makedirs(STACKING_DIR, exist_ok=True)

# 📦 Classes
CLASS_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
CLASS_TO_INDEX = {cls: i for i, cls in enumerate(CLASS_NAMES)}

# 🔧 Image preprocessing
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_preprocessed = preprocess_input(np.expand_dims(img_array, axis=0))
    return img_preprocessed

# 🧠 Load CNN model
cnn_model = load_model(MODEL_PATH)

# 🔄 Process images to get probabilities
def extract_probs_from_dir(dir_path):
    image_ids, probs, true_labels = [], [], []
    for label in os.listdir(dir_path):
        label_folder = os.path.join(dir_path, label)
        if not os.path.isdir(label_folder):
            continue
        for img_file in tqdm(os.listdir(label_folder), desc=f"Processing {label}"):
            img_path = os.path.join(label_folder, img_file)
            img_id = os.path.splitext(img_file)[0]
            img_input = preprocess_image(img_path)
            pred = cnn_model.predict(img_input, verbose=0)[0]
            image_ids.append(img_id)
            probs.append(pred)
            true_labels.append(label)
    return pd.DataFrame({"image_id": image_ids, "true_label": true_labels, **{f"proba_{cls}": [p[i] for p in probs] for i, cls in enumerate(CLASS_NAMES)}})

# 📄 Load metadata
metadata = pd.read_csv(METADATA_PATH)

# 🧪 Générer les features CNN pour train + val
train_df = extract_probs_from_dir(TRAIN_DIR)
val_df = extract_probs_from_dir(VAL_DIR)

train_df = train_df.merge(metadata, on="image_id")
val_df = val_df.merge(metadata, on="image_id")

# 🔧 Encodage pipeline
categorical = ["sex", "localization"]
numerical = ["age"]
proba_features = [f"proba_{cls}" for cls in CLASS_NAMES]

preprocessor = ColumnTransformer([
    ("proba", StandardScaler(), proba_features),
    ("num", Pipeline([("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler())]), numerical),
    ("cat", OneHotEncoder(), categorical)
])

# 🎯 Pipeline
X_train = pd.concat([train_df[proba_features], train_df[categorical + numerical]], axis=1)
y_train = train_df["true_label"]
X_val = pd.concat([val_df[proba_features], val_df[categorical + numerical]], axis=1)
y_val = val_df["true_label"]

pipeline = Pipeline([
    ("preproc", preprocessor),
    ("clf", RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42))
])

# 🏋️ Entraînement
pipeline.fit(X_train, y_train)

# 🔍 Évaluation
print("\n✅ Rapport de classification - Modèle Stacking (CNN + tabulaire) :")
y_pred = pipeline.predict(X_val)
report = classification_report(y_val, y_pred, target_names=CLASS_NAMES, output_dict=False)
print(report)

with open(os.path.join(STACKING_DIR, "classification_report_stacking.txt"), "w") as f:
    f.write(report)

# 🔲 Confusion matrix
cm = confusion_matrix(y_val, y_pred, labels=CLASS_NAMES, normalize='true') * 100
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt=".1f", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, cmap="Blues")
plt.title("Confusion Matrix (%) - Stacking Model")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig(os.path.join(STACKING_DIR, "confusion_matrix_stacking.png"))
plt.close()

# 🆚 Comparatif : CNN seul
y_val_indices = [CLASS_TO_INDEX[label] for label in y_val]
y_cnn_pred = val_df[proba_features].values.argmax(axis=1)
cnn_report = classification_report(y_val_indices, y_cnn_pred, target_names=CLASS_NAMES, output_dict=False)
print("\n✅ Rapport de classification - CNN seul (baseline) :")
print(cnn_report)

with open(os.path.join(STACKING_DIR, "classification_report_cnn.txt"), "w") as f:
    f.write(cnn_report)

# 💾 Sauvegarde du modèle empilé
joblib.dump(pipeline, os.path.join(STACKING_DIR, "stacked_model_rf.joblib"))
print("\n✅ Modèle empilé sauvegardé avec succès.")
