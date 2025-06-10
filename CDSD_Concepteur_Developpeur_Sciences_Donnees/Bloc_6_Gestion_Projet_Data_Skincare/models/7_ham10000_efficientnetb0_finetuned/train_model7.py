import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight

# Parser
parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--batch_size", type=int, default=32)
args = parser.parse_args()

# Paths
current_dir = os.path.abspath(os.path.dirname(__file__))
train_dir = os.path.abspath(os.path.join(current_dir, "..", "..", "src_pictures", "train1_ham10000"))
val_dir = os.path.abspath(os.path.join(current_dir, "..", "..", "src_pictures", "val1_ham10000"))
output_dir = current_dir
os.makedirs(output_dir, exist_ok=True)

# Data generators
train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_gen = train_datagen.flow_from_directory(
    train_dir, target_size=(224, 224), batch_size=args.batch_size, class_mode='categorical'
)
val_gen = val_datagen.flow_from_directory(
    val_dir, target_size=(224, 224), batch_size=args.batch_size, class_mode='categorical', shuffle=False
)

# Class weights
y_train = train_gen.classes
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights = dict(enumerate(class_weights))

# Model with fine-tuning
base_model = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
for layer in base_model.layers[:-20]:
    layer.trainable = False
for layer in base_model.layers[-20:]:
    layer.trainable = True

x = GlobalAveragePooling2D()(base_model.output)
output = Dense(train_gen.num_classes, activation="softmax")(x)
model = Model(inputs=base_model.input, outputs=output)
model.compile(optimizer=Adam(learning_rate=1e-4), loss="categorical_crossentropy", metrics=["accuracy"])

# Callbacks
checkpoint_path = os.path.join(output_dir, "model7_bestmodel.h5")
callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True),
    ReduceLROnPlateau(patience=3, factor=0.2, verbose=1),
    ModelCheckpoint(checkpoint_path, save_best_only=True)
]

# Training
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=args.epochs,
    class_weight=class_weights,
    callbacks=callbacks
)

# Plotting curves
plt.figure()
plt.plot(history.history["accuracy"], label="Train acc")
plt.plot(history.history["val_accuracy"], label="Val acc")
plt.legend()
plt.title("Accuracy")
plt.savefig(os.path.join(output_dir, "model7_accuracy.png"))

plt.figure()
plt.plot(history.history["loss"], label="Train loss")
plt.plot(history.history["val_loss"], label="Val loss")
plt.legend()
plt.title("Loss")
plt.savefig(os.path.join(output_dir, "model7_loss.png"))

# Evaluation
y_true = val_gen.classes
y_pred_probs = model.predict(val_gen)
y_pred = np.argmax(y_pred_probs, axis=1)

# Classification report
report = classification_report(y_true, y_pred, target_names=list(val_gen.class_indices.keys()))
with open(os.path.join(output_dir, "model7_classification_report.txt"), "w") as f:
    f.write(report)

# Confusion matrix in percentage
cm = confusion_matrix(y_true, y_pred)
cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

plt.figure(figsize=(10, 8))
sns.heatmap(cm_percent, annot=True, fmt=".1f", cmap="Blues",
            xticklabels=val_gen.class_indices.keys(),
            yticklabels=val_gen.class_indices.keys())
plt.title("Confusion Matrix (%)")
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.savefig(os.path.join(output_dir, "model7_confusion_matrix_percent.png"))
