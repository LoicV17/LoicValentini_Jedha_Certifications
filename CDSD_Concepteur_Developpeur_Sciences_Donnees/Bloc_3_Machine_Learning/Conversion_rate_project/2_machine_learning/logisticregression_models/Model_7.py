# Import basic libraries and appropriate scikit-learn libraries

import pandas as pd
import numpy as np



from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix, recall_score, precision_score, ConfusionMatrixDisplay, RocCurveDisplay)

import time
import mlflow
import mlflow.sklearn

from preprocessings.Preprocessing_4 import X_train, X_test, Y_train, Y_test, categorical_features, numeric_features

# To define the start time
start_time = time.time()


# Grid of values to be tested
params = {
    "C": [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 0.8, 1, 1.2, 1.5, 2, 5, 10, 20, 50, 100],
    "penalty" : ['l1','l2'],
    "solver": ['liblinear']
}
gridsearch = GridSearchCV(
    LogisticRegression(), param_grid=params, cv=5, scoring='f1'
)  # cv : the number of folds to be used for CV
gridsearch.fit(X_train, Y_train)
print()
print("Best hyperparameters : ", gridsearch.best_params_)
print("Best validation f1 : ", gridsearch.best_score_)


# MLflow run
with mlflow.start_run(run_name="model7"):
    # Model training
    model7 = gridsearch.best_estimator_
    model7.fit(X_train, Y_train)
    
    # Predictions on train set and test set and setting with new probability
    Y_train_pred = model7.predict(X_train)
    Y_train_pred_proba = model7.predict_proba(X_train)[:, 1]
    Y_train_pred = (Y_train_pred_proba >= 0.38).astype(int)

    Y_test_pred = model7.predict(X_test)
    Y_test_pred_proba = model7.predict_proba(X_test)[:, 1]
    Y_test_pred = (Y_test_pred_proba >= 0.38).astype(int)

    # Log the model and parameters
    mlflow.sklearn.log_model(model7, "model")
    mlflow.log_param("model", "logistic_regression_multivariate")
    mlflow.log_param("preprocessing", "preprocessing_4")
    mlflow.log_param("features", "age_without_outliers, country, new_user, total_pages_visited")
    mlflow.log_param("decision_threshold", 0.38)
    mlflow.log_param("penalty", gridsearch.best_params_["penalty"])
    mlflow.log_param("C", gridsearch.best_params_["C"])
    mlflow.log_param("solver", gridsearch.best_params_["solver"])


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



