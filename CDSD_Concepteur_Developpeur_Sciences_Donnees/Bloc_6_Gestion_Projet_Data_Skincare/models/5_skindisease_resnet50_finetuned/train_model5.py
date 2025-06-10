import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import pandas as pd

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=40)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--unfreeze_layers', type=int, default=30)
args = parser.parse_args()

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(BASE_DIR, '..', '..', 'src_pictures', 'train_skindisease')
VAL_DIR = os.path.join(BASE_DIR, '..', '..', 'src_pictures', 'val_skindisease')
TEST_DIR = os.path.join(BASE_DIR, '..', '..', 'src_pictures', 'test_skindisease')
MODEL_PATH = os.path.join(BASE_DIR, 'model5_best_model.h5')

# Image generators
datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
train_gen = datagen.flow_from_directory(TRAIN_DIR, target_size=(224, 224), batch_size=args.batch_size, class_mode='binary')
val_gen = datagen.flow_from_directory(VAL_DIR, target_size=(224, 224), batch_size=args.batch_size, class_mode='binary')
test_gen = datagen.flow_from_directory(TEST_DIR, target_size=(224, 224), batch_size=1, class_mode='binary', shuffle=False)

# Model definition
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
for layer in base_model.layers[:-args.unfreeze_layers]:
    layer.trainable = False
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(1, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=output)

model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])

# Callbacks
callbacks = [
    EarlyStopping(monitor='val_auc', patience=5, mode='max', restore_best_weights=True),
    ModelCheckpoint(MODEL_PATH, monitor='val_auc', mode='max', save_best_only=True)
]

# Training
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=args.epochs,
    callbacks=callbacks,
    verbose=1
)

# Plotting
plt.figure()
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy')
plt.legend()
plt.savefig(os.path.join(BASE_DIR, 'model5_accuracy.png'))

plt.figure()
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.legend()
plt.savefig(os.path.join(BASE_DIR, 'model5_loss.png'))

# Evaluation
preds = model.predict(test_gen)
preds_classes = (preds > 0.5).astype(int).flatten()
true_classes = test_gen.classes
labels = list(test_gen.class_indices.keys())

report = classification_report(true_classes, preds_classes, target_names=labels, output_dict=True)
pd.DataFrame(report).transpose().to_csv(os.path.join(BASE_DIR, 'model5_classification_report.csv'))

# Confusion Matrix
cm = confusion_matrix(true_classes, preds_classes)
sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Blues')
plt.title('Confusion Matrix')
plt.savefig(os.path.join(BASE_DIR, 'model5_confusion_matrix.png'))
print("✅ Model 5 ResNet50 fine-tuned trained and evaluated.")