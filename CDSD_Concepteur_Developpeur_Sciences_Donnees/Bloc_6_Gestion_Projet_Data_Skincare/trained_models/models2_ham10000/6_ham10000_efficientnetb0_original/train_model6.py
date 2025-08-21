import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import plot_model

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--batch_size", type=int, default=32)
args = parser.parse_args()

# Corrected paths (Windows-friendly and compatible with script location)
train_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src_pictures", "train1_ham10000"))
val_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src_pictures", "val1_ham10000"))
output_dir = os.path.abspath(os.path.dirname(__file__))


# Crée le dossier courant s'il n'existe pas (sécurité)
os.makedirs(output_dir, exist_ok=True)

datagen_train = ImageDataGenerator(preprocessing_function=preprocess_input)
datagen_val = ImageDataGenerator(preprocessing_function=preprocess_input)


train_gen = datagen_train.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=args.batch_size,
    class_mode='categorical'
)

val_gen = datagen_val.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=args.batch_size,
    class_mode='categorical',
    shuffle=False
)

# Class weights
from sklearn.utils import class_weight
y_train = train_gen.classes
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights = dict(enumerate(class_weights))

# Model
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze base

x = GlobalAveragePooling2D()(base_model.output)
output = Dense(train_gen.num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=output)

model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks
checkpoint_path = os.path.join(output_dir, "model6_bestmodel.h5")
callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True),
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

# Plotting
plt.figure()
plt.plot(history.history['accuracy'], label="Train acc")
plt.plot(history.history['val_accuracy'], label="Val acc")
plt.title("Accuracy")
plt.legend()
plt.savefig(os.path.join(output_dir, "model6_accuracy.png"))

plt.figure()
plt.plot(history.history['loss'], label="Train loss")
plt.plot(history.history['val_loss'], label="Val loss")
plt.title("Loss")
plt.legend()
plt.savefig(os.path.join(output_dir, "model6_loss.png"))

# Evaluation
val_gen.reset()
y_true = val_gen.classes
y_pred = model.predict(val_gen, verbose=1)
y_pred_classes = np.argmax(y_pred, axis=1)

# Classification report
report = classification_report(y_true, y_pred_classes, target_names=list(val_gen.class_indices.keys()))
with open(os.path.join(output_dir, "model6_classification_report.txt"), "w") as f:
    f.write(report)

# Confusion matrix in percentage
cm = confusion_matrix(y_true, y_pred_classes)
cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

plt.figure(figsize=(10, 8))
sns.heatmap(cm_percent, annot=True, fmt=".1f", cmap="Blues",
            xticklabels=val_gen.class_indices.keys(),
            yticklabels=val_gen.class_indices.keys())
plt.title("Confusion Matrix (%)")
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.savefig(os.path.join(output_dir, "model6_confusion_matrix_percent.png"))
