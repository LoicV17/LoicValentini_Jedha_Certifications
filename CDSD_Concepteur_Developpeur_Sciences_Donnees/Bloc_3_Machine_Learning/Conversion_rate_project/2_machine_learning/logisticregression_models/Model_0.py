# Import basic libraries and appropriate scikit-learn libraries

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix, recall_score, precision_score, ConfusionMatrixDisplay, RocCurveDisplay)

import time



data = pd.read_csv('../src/conversion_data_train.csv')
print("Data successfully retrieved")

# To define the start time
start_time = time.time()

# Only one feature saved, univariate model

features_list = ['total_pages_visited']
numeric_features = ['total_pages_visited']
target_variable = 'converted'

# Separation of X (features) and Y (target)
X = data.loc[:, features_list]
Y = data.loc[:, target_variable]
print()
print("X and Y successfully created")

# Dividing into train_set and test_set 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0, stratify=Y)
print()
print("Train-test split successfully done")

# Standardizing the numeric feature on training data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
print()
print("Training data successfully standardized")

# Transform only on test data
X_test = scaler.transform(X_test)
print()
print("Test data successfully standardized")

# Logistic regression model to train

model0 = LogisticRegression()
model0.fit(X_train, Y_train)
print()
print("Baseline model successfully trained")

import mlflow
import mlflow.sklearn

mlflow.set_experiment("conversion_rate_challenge")

with mlflow.start_run(run_name="baseline_model"):
    model0.fit(X_train, Y_train)
    Y_train_pred = model0.predict(X_train)
    Y_test_pred = model0.predict(X_test)

    # Log the model and parameters
    mlflow.sklearn.log_model(model0, "model")
    mlflow.log_param("model", "logistic_regression_univariate")
    mlflow.log_param("features", "total_pages_visited")

    # Metrics
    mlflow.log_metric("f1_train", round(f1_score(Y_train, Y_train_pred),3))
    mlflow.log_metric("f1_test", round(f1_score(Y_test, Y_test_pred),3))
    mlflow.log_metric("accuracy_train", round(accuracy_score(Y_train, Y_train_pred),3))
    mlflow.log_metric("accuracy_test", round(accuracy_score(Y_test, Y_test_pred),3))
    mlflow.log_metric("precision_train", round(precision_score(Y_train, Y_train_pred),3))
    mlflow.log_metric("precision_test", round(precision_score(Y_test, Y_test_pred),3))
    mlflow.log_metric("recall_train", round(recall_score(Y_train, Y_train_pred),3))
    mlflow.log_metric("recall_test", round(recall_score(Y_test, Y_test_pred),3))


# Metrics

print("F1-score on train set: ", f1_score(Y_train, Y_train_pred))
print("F1-score on test set: ", f1_score(Y_test, Y_test_pred))

# To define the end time
end_time = time.time()

# To define the execution time
execution_time = end_time - start_time





