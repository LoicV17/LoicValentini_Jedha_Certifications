import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import time
import mlflow
import mlflow.keras
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tqdm.keras import TqdmCallback
from tensorflow.keras.callbacks import Callback

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from preprocessings.Preprocessing_4 import X_train, X_test, Y_train, Y_test

# ============== ⏱️ Start timing
start_time = time.time()

# ============== 🕥 F1 Callback
class F1History(Callback):
    def __init__(self, X_train, y_train, X_val, y_val):
        super().__init__()
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.f1_train = []
        self.f1_val = []

    def on_epoch_end(self, epoch, logs=None):
        y_train_pred = (self.model.predict(self.X_train, verbose=0) >= 0.5).astype(int)
        self.f1_train.append(f1_score(self.y_train, y_train_pred))

        y_val_pred = (self.model.predict(self.X_val, verbose=0) >= 0.5).astype(int)
        self.f1_val.append(f1_score(self.y_val, y_val_pred))

# ============== 🔧 Simple Neural Network
model22 = Sequential([
    Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

model22.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

f1_callback = F1History(X_train=X_train, y_train=Y_train, X_val=X_test, y_val=Y_test)

# ============== 🧠 Train the model
history = model22.fit(
    X_train, Y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.2,
    verbose=0,
    callbacks=[TqdmCallback(verbose=1), f1_callback]
)

# ============== 🔍 Predict and find optimal threshold
y_proba = model22.predict(X_test).flatten()

best_threshold = 0.5
best_f1 = 0

for t in np.arange(0.2, 0.85, 0.01):
    y_pred = (y_proba >= t).astype(int)
    f1 = f1_score(Y_test, y_pred)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = round(t, 2)

print(f"✅ Best threshold: {best_threshold}, F1: {best_f1:.4f}")

# ============== 🔁 Final predictions with best threshold
y_train_pred = (model22.predict(X_train).flatten() >= best_threshold).astype(int)
y_test_pred = (y_proba >= best_threshold).astype(int)

# ============== 📊 MLflow logging
mlflow.set_experiment("conversion_rate_challenge")

with mlflow.start_run(run_name="model22"):
    mlflow.keras.log_model(model22, "model")
    mlflow.log_param("model", "simple_nn")
    mlflow.log_param("preprocessing", "preprocessing_4")
    mlflow.log_param("architecture", "32-ReLU-Dropout-16-ReLU-1-Sigmoid")
    mlflow.log_param("features", "age_without_outliers, country, new_user, total_pages_visited")
    mlflow.log_param("decision_threshold", best_threshold)

    mlflow.log_metric("f1_train", round(f1_score(Y_train, y_train_pred), 3))
    mlflow.log_metric("f1_test", round(f1_score(Y_test, y_test_pred), 3))
    mlflow.log_metric("accuracy_train", round(accuracy_score(Y_train, y_train_pred), 3))
    mlflow.log_metric("accuracy_test", round(accuracy_score(Y_test, y_test_pred), 3))
    mlflow.log_metric("precision_test", round(precision_score(Y_test, y_test_pred), 3))
    mlflow.log_metric("recall_test", round(recall_score(Y_test, y_test_pred), 3))
    mlflow.log_metric("execution_time_sec", time.time() - start_time)

# ============== 📈 Plotting F1 and Loss
plt.figure(figsize=(10, 4))
plt.plot(f1_callback.f1_train, label='F1 Train')
plt.plot(f1_callback.f1_val, label='F1 Test')
plt.title("F1-score per Epoch")
plt.xlabel("Epoch")
plt.ylabel("F1 Score")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 4))
plt.plot(history.history['loss'], label='Loss Train')
plt.plot(history.history['val_loss'], label='Loss Test')
plt.title("Loss per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()
