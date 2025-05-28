import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix, recall_score, precision_score, ConfusionMatrixDisplay, RocCurveDisplay)
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

import time
import mlflow
import mlflow.sklearn

from preprocessings.Preprocessing_4 import X_train, X_test, Y_train, Y_test, categorical_features, numeric_features

# To define the start time
start_time = time.time()

# Base estimator: logistic regression
base_model = LogisticRegression(max_iter=1000)

# Paramètres à tester dans GridSearch
param_grid = {
    'adaboostclassifier__n_estimators': [10, 20, 50, 100, 200],
    'adaboostclassifier__learning_rate': [0.01, 0.05, 0.1, 0.5, 1.0]
}

# Pipeline AdaBoost + LogisticRegression
pipeline = make_pipeline(
    AdaBoostClassifier(estimator=base_model, algorithm='SAMME', random_state=0)
)


# GridSearch avec validation croisée
grid = GridSearchCV(pipeline, param_grid=param_grid, scoring='f1', cv=5, verbose=1, n_jobs=-1)
grid.fit(X_train, Y_train)

# Résultat
print("Best params:", grid.best_params_)
print("Best F1 score:", grid.best_score_)


# MLflow run
with mlflow.start_run(run_name="model15"):
    # Model training
    model15 = grid.best_estimator_
    model15.fit(X_train, Y_train)
    
    # Predictions on train set and test set and setting with new probability
    Y_train_pred = model15.predict(X_train)
    Y_train_pred_proba = model15.predict_proba(X_train)[:, 1]
    Y_train_pred = (Y_train_pred_proba >= 0.38).astype(int)

    Y_test_pred = model15.predict(X_test)
    Y_test_pred_proba = model15.predict_proba(X_test)[:, 1]
    Y_test_pred = (Y_test_pred_proba >= 0.38).astype(int)

    # Log the model and parameters
    mlflow.sklearn.log_model(model15, "model")
    mlflow.log_param("model", "adaboost_logistic_regression")
    mlflow.log_param("preprocessing", "preprocessing_4")
    mlflow.log_param("features", "age_without_outliers, country, new_user, total_pages_visited")
    mlflow.log_param("decision_threshold", 0.38)
    mlflow.log_param('adaboostclassifier__n_estimators', grid.best_params_['adaboostclassifier__n_estimators'])
    mlflow.log_param('adaboostclassifier__learning_rate', grid.best_params_['adaboostclassifier__learning_rate'])


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

