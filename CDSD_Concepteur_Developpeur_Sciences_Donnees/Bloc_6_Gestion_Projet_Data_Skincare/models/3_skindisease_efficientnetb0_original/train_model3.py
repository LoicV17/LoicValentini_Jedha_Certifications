import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=32)
args = parser.parse_args()

# 📁 Chemins
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(BASE_DIR, "src_pictures")
MODEL_DIR = os.path.join(BASE_DIR, "models", "3_skindisease_efficientnetb0_original")
os.makedirs(MODEL_DIR, exist_ok=True)

train_dir = os.path.join(DATA_DIR, "train_skindisease")
val_dir   = os.path.join(DATA_DIR, "val_skindisease")
test_dir  = os.path.join(DATA_DIR, "test_skindisease")

# 📦 Datasets
IMG_SIZE = (224, 224)

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    label_mode="binary",
    image_size=IMG_SIZE,
    batch_size=args.batch_size,
    shuffle=True
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    val_dir,
    label_mode="binary",
    image_size=IMG_SIZE,
    batch_size=args.batch_size,
    shuffle=False
)

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    test_dir,
    label_mode="binary",
    image_size=IMG_SIZE,
    batch_size=args.batch_size,
    shuffle=False
)

# Prétraitement
preprocess = lambda ds: ds.map(lambda x, y: (preprocess_input(x), y))
train_ds = preprocess(train_ds)
val_ds = preprocess(val_ds)
test_ds = preprocess(test_ds)

# ⚙️ EfficientNetB0 gelé
base_model = EfficientNetB0(include_top=False, weights="imagenet", input_shape=(224, 224, 3))
base_model.trainable = False

# Architecture
inputs = Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = GlobalAveragePooling2D()(x)
outputs = Dense(1, activation='sigmoid')(x)
model = Model(inputs, outputs)

# Compilation
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Callbacks
checkpoint_path = os.path.join(MODEL_DIR, "model3_bestmodel.h5")
callbacks = [
    ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='val_accuracy', mode='max'),
    EarlyStopping(patience=3, restore_best_weights=True)
]

# Entraînement
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=args.epochs,
    callbacks=callbacks,
    verbose=1
)

# Courbes
def plot_history(metric):
    plt.figure()
    plt.plot(history.history[metric], label=f"train_{metric}")
    plt.plot(history.history[f"val_{metric}"], label=f"val_{metric}")
    plt.title(f"{metric.capitalize()} Evolution")
    plt.xlabel("Epoch")
    plt.ylabel(metric.capitalize())
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(MODEL_DIR, f"{metric}_curve.png"))
    plt.close()

plot_history("accuracy")
plot_history("loss")

# Évaluation
y_true, y_pred = [], []
for images, labels in test_ds:
    preds = model.predict(images)
    y_true.extend(labels.numpy())
    y_pred.extend(preds.flatten())

y_pred_bin = [1 if p >= 0.5 else 0 for p in y_pred]

# Rapport
report = classification_report(y_true, y_pred_bin, target_names=["benign", "malignant"])
with open(os.path.join(MODEL_DIR, "classification_report.txt"), "w") as f:
    f.write(report)

# Matrice de confusion
cm = confusion_matrix(y_true, y_pred_bin)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["benign", "malignant"], yticklabels=["benign", "malignant"])
plt.title("Matrice de confusion")
plt.xlabel("Prédiction")
plt.ylabel("Réel")
plt.savefig(os.path.join(MODEL_DIR, "confusion_matrix.png"))
plt.close()

print("✅ Modèle EfficientNetB0 gelé entraîné et évalué.")
