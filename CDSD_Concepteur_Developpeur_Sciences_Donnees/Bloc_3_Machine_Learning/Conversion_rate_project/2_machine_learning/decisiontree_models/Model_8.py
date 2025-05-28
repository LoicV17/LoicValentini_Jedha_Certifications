# Import basic libraries and appropriate scikit-learn libraries

import pandas as pd
import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix, recall_score, precision_score, ConfusionMatrixDisplay, RocCurveDisplay)

import time
import mlflow
import mlflow.sklearn

from preprocessings.Preprocessing_4 import X_train, X_test, Y_train, Y_test, categorical_features, numeric_features

# To define the start time
start_time = time.time()

# Grid of values to be tested
params = {
    "max_depth": [4, 6, 8, 10, 15, 20],
    "min_samples_leaf": [1, 2, 5],
    "min_samples_split": [2, 4, 8],
}

gridsearch = GridSearchCV(
    DecisionTreeClassifier(), param_grid=params, cv=5
)  # cv : the number of folds to be used for CV
gridsearch.fit(X_train, Y_train)
print("Best hyperparameters : ", gridsearch.best_params_)
print("Best validation f1 : ", gridsearch.best_score_)


# MLflow run
with mlflow.start_run(run_name="model8"):
    # Model training
    model8 = gridsearch.best_estimator_
    model8.fit(X_train, Y_train)
    
    # Predictions on train set and test set and setting with new probability
    Y_train_pred = model8.predict(X_train)
    Y_train_pred_proba = model8.predict_proba(X_train)[:, 1]
    Y_train_pred = (Y_train_pred_proba >= 0.38).astype(int)

    Y_test_pred = model8.predict(X_test)
    Y_test_pred_proba = model8.predict_proba(X_test)[:, 1]
    Y_test_pred = (Y_test_pred_proba >= 0.38).astype(int)

    # Log the model and parameters
    mlflow.sklearn.log_model(model8, "model")
    mlflow.log_param("model", "decision_tree")
    mlflow.log_param("preprocessing", "preprocessing_4")
    mlflow.log_param("features", "age_without_outliers, country, new_user, total_pages_visited")
    mlflow.log_param("decision_threshold", 0.38)
    mlflow.log_param("max_depth", gridsearch.best_params_["max_depth"])
    mlflow.log_param("min_samples_leaf", gridsearch.best_params_["min_samples_leaf"])
    mlflow.log_param("min_samples_split", gridsearch.best_params_["min_samples_split"])



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



