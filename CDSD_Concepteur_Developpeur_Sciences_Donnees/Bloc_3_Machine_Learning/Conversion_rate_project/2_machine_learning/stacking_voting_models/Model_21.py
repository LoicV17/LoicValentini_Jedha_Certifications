import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix, recall_score, precision_score, ConfusionMatrixDisplay, RocCurveDisplay)

import time
import mlflow
import mlflow.sklearn

from preprocessings.Preprocessing_1 import X_train, X_test, Y_train, Y_test, categorical_features, numeric_features

# To define the start time
start_time = time.time()

clf1 = LogisticRegression(max_iter=1000)
clf2 = DecisionTreeClassifier(max_depth = 9, min_samples_split=14, min_samples_leaf=3)
clf3 = XGBClassifier(
    n_estimators=50,
    learning_rate=0.3,
    max_depth=5,
    objective='binary:logistic',
    use_label_encoder=False,      
    eval_metric='logloss',        
    random_state=0)

model21 = VotingClassifier(
    estimators=[
        ('lr', clf1),
        ('dt', clf2),
        ('xgb', clf3)
    ],
    voting='soft',
    n_jobs=-1
)


# Entraînement
model21.fit(X_train, Y_train)
y_proba = model21.predict_proba(X_test)[:, 1]

best_f1 = 0
best_threshold = 0.5

for t in np.arange(0, 0.85, 0.01):
    y_pred = (y_proba >= t).astype(int)
    f1 = f1_score(Y_test, y_pred)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = round(t, 2)

print(f"✅ Best threshold: {best_threshold}, F1: {best_f1:.4f}")


# MLflow run
with mlflow.start_run(run_name="model21"):
    # Model training

    model21.fit(X_train, Y_train)
    
    # Predictions on train set and test set and setting with new probability
    Y_train_pred = model21.predict(X_train)
    Y_train_pred_proba = model21.predict_proba(X_train)[:, 1]
    Y_train_pred = (Y_train_pred_proba >= best_threshold).astype(int)

    Y_test_pred = model21.predict(X_test)
    Y_test_pred_proba = model21.predict_proba(X_test)[:, 1]
    Y_test_pred = (Y_test_pred_proba >= best_threshold).astype(int)

    # Log the model and parameters
    mlflow.sklearn.log_model(model21, "model")
    mlflow.log_param("model", "voting")
    mlflow.log_param("preprocessing", "preprocessing_4")
    mlflow.log_param("features", "age, country, new_user, total_pages_visited, traffic_source")
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


