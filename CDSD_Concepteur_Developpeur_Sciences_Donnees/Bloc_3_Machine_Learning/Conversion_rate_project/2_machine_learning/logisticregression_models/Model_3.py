import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import appropriate scikit-learn libraries

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, recall_score,
    precision_score, ConfusionMatrixDisplay, RocCurveDisplay
)

import time
import mlflow
import mlflow.sklearn

from preprocessings.Preprocessing_3 import (
    X_train, X_test, Y_train, Y_test, categorical_features, numeric_features
)


# Set MLflow experiment
mlflow.set_experiment("conversion_rate_challenge")

# Start time
start_time = time.time()

# MLflow run
with mlflow.start_run(run_name="model3"):
    # Model training
    model3 = LogisticRegression()
    model3.fit(X_train, Y_train)
    Y_train_pred = model3.predict(X_train)
    Y_test_pred = model3.predict(X_test)

    # Log the model and parameters
    mlflow.sklearn.log_model(model3, "model")
    mlflow.log_param("model", "logistic_regression_multivariate")
    mlflow.log_param("preprocessing", "preprocessing_3")
    mlflow.log_param("features", "age_without_outliers, country, new_user, total_pages_visited")

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
