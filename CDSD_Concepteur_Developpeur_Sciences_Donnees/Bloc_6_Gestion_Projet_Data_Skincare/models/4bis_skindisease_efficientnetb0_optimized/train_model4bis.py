import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=40)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--unfreeze_layers', type=int, default=20)
args = parser.parse_args()

# Paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(BASE_DIR, "src_pictures")
MODEL_DIR = os.path.join(BASE_DIR, "models", "4bis_skindisease_efficientnetb0_optimized")
os.makedirs(MODEL_DIR, exist_ok=True)

train_dir = os.path.join(DATA_DIR, "train_skindisease")
val_dir   = os.path.join(DATA_DIR, "val_skindisease")
test_dir  = os.path.join(DATA_DIR, "test_skindisease")

# Dataset settings
IMG_SIZE = (240, 240)
AUTOTUNE = tf.data.AUTOTUNE

# Data augmentation
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomContrast(0.1),
])

# Load datasets
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir, label_mode="binary", image_size=IMG_SIZE, batch_size=args.batch_size, shuffle=True)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    val_dir, label_mode="binary", image_size=IMG_SIZE, batch_size=args.batch_size, shuffle=False)
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    test_dir, label_mode="binary", image_size=IMG_SIZE, batch_size=args.batch_size, shuffle=False)

# Preprocess pipeline
train_ds = train_ds.map(lambda x, y: (data_augmentation(x), y), num_parallel_calls=AUTOTUNE)
train_ds = train_ds.map(lambda x, y: (preprocess_input(x), y), num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
val_ds = val_ds.map(lambda x, y: (preprocess_input(x), y), num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
test_ds = test_ds.map(lambda x, y: (preprocess_input(x), y), num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

# Base model setup
base_model = EfficientNetB0(include_top=False, weights="imagenet", input_shape=(240, 240, 3))
base_model.trainable = True
for layer in base_model.layers[:-args.unfreeze_layers]:
    layer.trainable = False

# Final model
inputs = Input(shape=(240, 240, 3))
x = base_model(inputs, training=True)
x = GlobalAveragePooling2D()(x)
outputs = Dense(1, activation='sigmoid')(x)
model = Model(inputs, outputs)

# Compile
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
)

# Callbacks
checkpoint_path = os.path.join(MODEL_DIR, "model4_best_model_v2.h5")
callbacks = [
    ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='val_accuracy', mode='max'),
    EarlyStopping(patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, verbose=1)
]

# Training
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=args.epochs,
    callbacks=callbacks,
    verbose=1
)

# Plot function
def plot_history(metric):
    plt.figure()
    plt.plot(history.history[metric], label=f"train_{metric}")
    plt.plot(history.history[f"val_{metric}"], label=f"val_{metric}")
    plt.title(f"{metric.capitalize()} Evolution")
    plt.xlabel("Epoch")
    plt.ylabel(metric.capitalize())
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(MODEL_DIR, f"{metric}_curve_v2.png"))
    plt.close()

plot_history("accuracy")
plot_history("loss")
plot_history("auc")

# Evaluation
y_true, y_pred = [], []
for images, labels in test_ds:
    preds = model.predict(images)
    y_true.extend(labels.numpy())
    y_pred.extend(preds.flatten())

y_pred_bin = [1 if p >= 0.5 else 0 for p in y_pred]

# Report
report = classification_report(y_true, y_pred_bin, target_names=["benign", "malignant"])
with open(os.path.join(MODEL_DIR, "classification_report_v2.txt"), "w") as f:
    f.write(report)

# Confusion matrix
cm = confusion_matrix(y_true, y_pred_bin)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["benign", "malignant"], yticklabels=["benign", "malignant"])
plt.title("Confusion Matrix (v2)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig(os.path.join(MODEL_DIR, "confusion_matrix_v2.png"))
plt.close()

print("\n✅ Fine-tuning EfficientNetB0 v2 terminé. Résultats enregistrés dans :", MODEL_DIR)
