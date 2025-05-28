import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import time
import mlflow
import mlflow.sklearn

from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score, precision_score
)

from preprocessings.Preprocessing_4 import X_train, X_test, Y_train, Y_test

# Start timer
start_time = time.time()

# Param grid for XGBoost
param_grid = {
    'n_estimators': [20, 50, 100, 200],
    'learning_rate': [0.01, 0.05, 0.1, 0.3],
    'max_depth': [3, 5, 7]
}

# GridSearchCV
grid = GridSearchCV(
    estimator=XGBClassifier(
        objective='binary:logistic',
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=0
    ),
    param_grid=param_grid,
    scoring='f1',
    cv=3,
    verbose=1,
    n_jobs=-1
)

grid.fit(X_train, Y_train)

# Résultats
print("Best params:", grid.best_params_)
print("Best F1 score:", grid.best_score_)

# === MLflow tracking ===
with mlflow.start_run(run_name="model16"):
    best_model = grid.best_estimator_
    best_model.fit(X_train, Y_train)

    # Predict with threshold 0.38
    Y_train_pred_proba = best_model.predict_proba(X_train)[:, 1]
    Y_train_pred = (Y_train_pred_proba >= 0.38).astype(int)

    Y_test_pred_proba = best_model.predict_proba(X_test)[:, 1]
    Y_test_pred = (Y_test_pred_proba >= 0.38).astype(int)

    # Log model + params
    mlflow.sklearn.log_model(best_model, "model")
    mlflow.log_param("model", "xgboost_classifier_logistic_regression")
    mlflow.log_param("preprocessing", "preprocessing_4")
    mlflow.log_param("features", "age_without_outliers, country, new_user, total_pages_visited")
    mlflow.log_param("decision_threshold", 0.38)
    mlflow.log_param("n_estimators", grid.best_params_["n_estimators"])
    mlflow.log_param("learning_rate", grid.best_params_["learning_rate"])
    mlflow.log_param("max_depth", grid.best_params_["max_depth"])

 
    # Log metrics
    mlflow.log_metric("f1_train", round(f1_score(Y_train, Y_train_pred), 3))
    mlflow.log_metric("f1_test", round(f1_score(Y_test, Y_test_pred), 3))
    mlflow.log_metric("accuracy_train", round(accuracy_score(Y_train, Y_train_pred), 3))
    mlflow.log_metric("accuracy_test", round(accuracy_score(Y_test, Y_test_pred), 3))
    mlflow.log_metric("precision_train", round(precision_score(Y_train, Y_train_pred), 3))
    mlflow.log_metric("precision_test", round(precision_score(Y_test, Y_test_pred), 3))
    mlflow.log_metric("recall_train", round(recall_score(Y_train, Y_train_pred), 3))
    mlflow.log_metric("recall_test", round(recall_score(Y_test, Y_test_pred), 3))

    mlflow.log_metric("execution_time_sec", round(time.time() - start_time, 2))
