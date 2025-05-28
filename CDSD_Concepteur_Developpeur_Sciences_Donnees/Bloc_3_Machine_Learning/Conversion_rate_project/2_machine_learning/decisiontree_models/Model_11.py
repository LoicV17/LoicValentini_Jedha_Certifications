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

# Objective function to maximise the F1-score
def objective(trial):

    # Hyperparameter search space
    param_grid = {
        'max_depth': trial.suggest_int('max_depth', 2, 100),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'random_state' : 0,
        'criterion' : 'entropy'
    }
    
    # Initialize the model with the suggested hyperparameters
    model = DecisionTreeClassifier(**param_grid)

    # Train the model
    model.fit(X_train, Y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Compute F1-score
    return f1_score(Y_test, y_pred)

# Create and run the study
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

best_params = study.best_params

# MLflow run
with mlflow.start_run(run_name="model11"):
    # Model training
    model11 = DecisionTreeClassifier(**best_params)
    model11.fit(X_train, Y_train)
    
    # Predictions on train set and test set and setting with new probability
    Y_train_pred = model11.predict(X_train)
    Y_train_pred_proba = model11.predict_proba(X_train)[:, 1]
    Y_train_pred = (Y_train_pred_proba >= 0.38).astype(int)

    Y_test_pred = model11.predict(X_test)
    Y_test_pred_proba = model11.predict_proba(X_test)[:, 1]
    Y_test_pred = (Y_test_pred_proba >= 0.38).astype(int)

    # Log the model and parameters
    mlflow.sklearn.log_model(model11, "model")
    mlflow.log_param("model", "decision_tree")
    mlflow.log_param("preprocessing", "preprocessing_4")
    mlflow.log_param("features", "age_without_outliers, country, new_user, total_pages_visited")
    mlflow.log_param("decision_threshold", 0.38)
    mlflow.log_param("max_depth", best_params["max_depth"])
    mlflow.log_param("min_samples_leaf", best_params["min_samples_leaf"])
    mlflow.log_param("min_samples_split", best_params["min_samples_split"])
    mlflow.log_param("criterion", "entropy")



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


