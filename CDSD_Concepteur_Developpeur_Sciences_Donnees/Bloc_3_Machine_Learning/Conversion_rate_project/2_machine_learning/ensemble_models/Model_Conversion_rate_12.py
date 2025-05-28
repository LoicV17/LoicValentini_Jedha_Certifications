# Import basic libraries and appropriate scikit-learn libraries

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix, recall_score, precision_score, ConfusionMatrixDisplay, RocCurveDisplay)
from sklearn.model_selection import GridSearchCV


import time

from Import_preprocessing_multivariate import X_train, X_test, Y_train, Y_test, categorical_features, numeric_features

# To define the start time
start_time = time.time()


# Base model
logistic_regression = LogisticRegression(max_iter=1000)

# Bagging model
model12_test = AdaBoostClassifier(estimator=logistic_regression)

# Grid of values to be tested
params = {"estimator__C": [0.8,0.9,1,1.1,1.2,1.3,1.4,1.5],
          "n_estimators": [5,10,20,30,50],
        }

print(params)
gridsearch = GridSearchCV(model12_test, param_grid=params, cv=3, scoring='f1', verbose=3)
gridsearch.fit(X_train, Y_train)

print()
print("Best hyperparameters : ", gridsearch.best_params_)
print("Best f1-score : ", gridsearch.best_score_)


# Recreate the model with the best parameters
model12 = AdaBoostClassifier(
    estimator=LogisticRegression(C=gridsearch.best_params_["estimator__C"], max_iter=1000),
    n_estimators=gridsearch.best_params_["n_estimators"]
)

model12.fit(X_train, Y_train)
print()
print("Model successfully trained")


# Predictions on train set and test set
Y_train_pred_proba = model12.predict_proba(X_train)[:, 1]
Y_train_pred = (Y_train_pred_proba >= 0.4).astype(int)

Y_test_pred_proba = model12.predict_proba(X_test)[:, 1]
Y_test_pred = (Y_test_pred_proba >= 0.4).astype(int)

print()
print("Predictions successfully made")



print()
print("F1-score on train set: ", f1_score(Y_train, Y_train_pred))
print("F1-score on test set: ", f1_score(Y_test, Y_test_pred))
print()
print("Accuracy on train set: ", accuracy_score(Y_train, Y_train_pred))
print("Accuracy on test set: ", accuracy_score(Y_test, Y_test_pred))
print()
print("Precision on train set: ", precision_score(Y_train, Y_train_pred))
print("Precision on test set: ", precision_score(Y_test, Y_test_pred))
print()
print("Recall on train set: ", recall_score(Y_train, Y_train_pred))
print("Recall on test set: ", recall_score(Y_test, Y_test_pred))
print()
print("Confusion matrix on train set: ")
print(confusion_matrix(Y_train, Y_train_pred))
print()
print("Confusion matrix on test set: ")
print(confusion_matrix(Y_test, Y_test_pred))


# To define the end time
end_time = time.time()

# To define the execution time
execution_time = end_time - start_time


model12_details = {
    'Model_ID' : 'model12',
    'Execution_time' : execution_time,
    'Model_Type' : 'LogisticRegression',
    'Multivariate' : 'Yes',
    'Categorical_features' : categorical_features,
    'Numeric_features' : numeric_features,
    'f1_train' : f1_score(Y_train, Y_train_pred),
    'f1_test' : f1_score(Y_test, Y_test_pred),
    'Accuracy_train' : accuracy_score(Y_train, Y_train_pred),
    'Accuracy_test' : accuracy_score(Y_test, Y_test_pred),
    'Precision_train' : precision_score(Y_train, Y_train_pred),
    'Precision_test' : precision_score(Y_test, Y_test_pred),
    'Recall_train' : recall_score(Y_train, Y_train_pred),
    'Recall_test' : recall_score(Y_test, Y_test_pred),
    'Decision_threshold' : 0.4,
    'Hyperparameter_C' : 1.2,
    'Hyperparameter_N_estimators' : gridsearch.best_params_["n_estimators"],
    'Hyperparameter_method' : 'GridsearchCV',
    'Boosting_type' : 'Adaboost'
}

