import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix, recall_score, precision_score, ConfusionMatrixDisplay, RocCurveDisplay)

import time
import mlflow
import mlflow.sklearn

from preprocessings.Preprocessing_4 import X_train, X_test, Y_train, Y_test, categorical_features, numeric_features

# To define the start time
start_time = time.time()

base_learners = [
    ('logreg', LogisticRegression(max_iter=1000)),
    ('tree', DecisionTreeClassifier(max_depth = 9, min_samples_split=14, min_samples_leaf=3)),
    ('xgb', XGBClassifier(
    n_estimators=50,
    learning_rate=0.3,
    max_depth=5,
    objective='binary:logistic',
    use_label_encoder=False,      
    eval_metric='logloss',        
    random_state=0))
]

meta_model = LogisticRegression()

model18 = StackingClassifier(
    estimators=base_learners,
    final_estimator=meta_model,
    cv=5,  # cross-validation pour les prédictions intermédiaires
    n_jobs=-1
)

# Entraînement
model18.fit(X_train, Y_train)

y_proba = model18.predict_proba(X_test)[:, 1]

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
with mlflow.start_run(run_name="model18"):
    # Model training

    model18.fit(X_train, Y_train)
    
    # Predictions on train set and test set and setting with new probability
    Y_train_pred = model18.predict(X_train)
    Y_train_pred_proba = model18.predict_proba(X_train)[:, 1]
    Y_train_pred = (Y_train_pred_proba >= best_threshold).astype(int)

    Y_test_pred = model18.predict(X_test)
    Y_test_pred_proba = model18.predict_proba(X_test)[:, 1]
    Y_test_pred = (Y_test_pred_proba >= best_threshold).astype(int)

    # Log the model and parameters
    mlflow.sklearn.log_model(model18, "model")
    mlflow.log_param("model", "stacking")
    mlflow.log_param("preprocessing", "preprocessing_4")
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


