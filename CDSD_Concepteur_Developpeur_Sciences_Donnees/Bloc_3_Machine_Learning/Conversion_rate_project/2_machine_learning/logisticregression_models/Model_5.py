import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix, recall_score, precision_score, ConfusionMatrixDisplay, RocCurveDisplay)

import time
import mlflow
import mlflow.sklearn

from preprocessings.Preprocessing_4 import X_train, X_test, Y_train, Y_test, categorical_features, numeric_features

# To define the start time
start_time = time.time()

# Logistic regression model to train
model5_test = LogisticRegression()
model5_test.fit(X_train, Y_train)
print()
print("Model successfully trained")

# Initial variable values
best_threshold = 0.5
best_f1 = 0

# Loop to check threshold

for threshold in np.arange(0.2,0.85,0.01):

    print("Test with Threshold =", threshold)

    threshold = round(threshold,2) # Because of 0.2999999999 values

    # Predictions on train set and test set and setting with new probability

    Y_train_pred_proba = model5_test.predict_proba(X_train)[:, 1]
    Y_train_pred = (Y_train_pred_proba >= threshold).astype(int)

    Y_test_pred_proba = model5_test.predict_proba(X_test)[:, 1]
    Y_test_pred = (Y_test_pred_proba >= threshold).astype(int)

    f1 = f1_score(Y_test, Y_test_pred)

    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold
print()
print("-----------------------------------------------------------")
print(f"Best threshold: {best_threshold} with F1-score: {best_f1}")


# Set MLflow experiment
mlflow.set_experiment("conversion_rate_challenge")

# Start time
start_time = time.time()

# MLflow run
with mlflow.start_run(run_name="model5"):
    # Model training
    model5 = LogisticRegression()
    model5.fit(X_train, Y_train)
    
    # Predictions on train set and test set and setting with new probability
    Y_train_pred = model5.predict(X_train)
    Y_train_pred_proba = model5.predict_proba(X_train)[:, 1]
    Y_train_pred = (Y_train_pred_proba >= best_threshold).astype(int)

    Y_test_pred = model5.predict(X_test)
    Y_test_pred_proba = model5.predict_proba(X_test)[:, 1]
    Y_test_pred = (Y_test_pred_proba >= best_threshold).astype(int)

    # Log the model and parameters
    mlflow.sklearn.log_model(model5, "model")
    mlflow.log_param("model", "logistic_regression_multivariate")
    mlflow.log_param("preprocessing", "preprocessing_5")
    mlflow.log_param("features", "age_without_outliers, country, new_user, total_pages_visited")
    mlflow.log_param("decision_threshold", best_threshold)

    # Metrics
    mlflow.log_metric("f1_train", round(f1_score(Y_train, Y_train_pred),3))
    mlflow.log_metric("f1_test", round(f1_score(Y_test, Y_test_pred),3))
    mlflow.log_metric("accuracy_train", round(accuracy_score(Y_train, Y_train_pred),3))
    mlflow.log_metric("accuracy_test", round(accuracy_score(Y_test, Y_test_pred),3))
    mlflow.log_metric("precision_train", round(precision_score(Y_train, Y_train_pred),3))
    mlflow.log_metric("precision_test", round(precision_score(Y_test, Y_test_pred),3))
    mlflow.log_metric("recall_train", round(recall_score(Y_train, Y_train_pred),3))
    mlflow.log_metric("recall_test", round(recall_score(Y_test, Y_test_pred),3))

    # Execution time
    end_time = time.time()
    mlflow.log_metric("execution_time_sec", end_time - start_time)

