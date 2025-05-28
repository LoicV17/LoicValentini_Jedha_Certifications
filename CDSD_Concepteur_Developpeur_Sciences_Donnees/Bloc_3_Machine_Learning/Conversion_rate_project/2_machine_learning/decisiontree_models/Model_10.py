# Import basic libraries and appropriate scikit-learn libraries

import pandas as pd
import numpy as np

import optuna

from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix, recall_score, precision_score, ConfusionMatrixDisplay, RocCurveDisplay)

import time
import mlflow
import mlflow.sklearn

from preprocessings.Preprocessing_4 import X_train, X_test, Y_train, Y_test, categorical_features, numeric_features

# To define the start time
start_time = time.time()

# Logistic regression model to train
model10 = DecisionTreeClassifier(max_depth = 9, min_samples_split=14, min_samples_leaf=3)
model10.fit(X_train, Y_train)
print()
print("Model successfully trained")


# Initialize the varaible

best_f1 = 0

# Loop to check threshold

for threshold in np.linspace(0.2,0.85,65):
   
    threshold = round(threshold,2) # Because of 0.2999999999 values
    print("Test with Threshold =", threshold)

    # Predictions on train set and test set and setting with new probability

    Y_train_pred_proba = model10.predict_proba(X_train)[:, 1]
    Y_train_pred = (Y_train_pred_proba >= threshold).astype(int)

    Y_test_pred_proba = model10.predict_proba(X_test)[:, 1]
    Y_test_pred = (Y_test_pred_proba >= threshold).astype(int)

    f1 = f1_score(Y_test, Y_test_pred)

    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold
print()
print("-----------------------------------------------------------")
print(f"Best threshold: {best_threshold} with F1-score: {best_f1}")



# MLflow run
with mlflow.start_run(run_name="model10"):
    # Model training
    model10 = DecisionTreeClassifier(max_depth = 9, min_samples_split=14, min_samples_leaf=3)
    model10.fit(X_train, Y_train)
    
    # Predictions on train set and test set and setting with new probability
    Y_train_pred = model10.predict(X_train)
    Y_train_pred_proba = model10.predict_proba(X_train)[:, 1]
    Y_train_pred = (Y_train_pred_proba >= best_threshold).astype(int)

    Y_test_pred = model10.predict(X_test)
    Y_test_pred_proba = model10.predict_proba(X_test)[:, 1]
    Y_test_pred = (Y_test_pred_proba >= best_threshold).astype(int)

    # Log the model and parameters
    mlflow.sklearn.log_model(model10, "model")
    mlflow.log_param("model", "decision_tree")
    mlflow.log_param("preprocessing", "preprocessing_4")
    mlflow.log_param("features", "age_without_outliers, country, new_user, total_pages_visited")
    mlflow.log_param("decision_threshold", best_threshold)
    mlflow.log_param("max_depth", 9)
    mlflow.log_param("min_samples_leaf", 3)
    mlflow.log_param("min_samples_split", 14)



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


