# Import basic libraries and appropriate scikit-learn libraries

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix, recall_score, precision_score, ConfusionMatrixDisplay, RocCurveDisplay)
from sklearn.model_selection import GridSearchCV


import time

from Import_preprocessing_multivariate import X_train, X_test, Y_train, Y_test, categorical_features, numeric_features

# To define the start time
start_time = time.time()

# Base model
decision_tree = DecisionTreeClassifier(max_depth = 10, min_samples_split=9, min_samples_leaf=8, criterion='entropy')

# Bagging model
model13 = AdaBoostClassifier(estimator=decision_tree)

model13.fit(X_train, Y_train)
print()
print("Model successfully trained")


# Predictions on train set and test set
Y_train_pred_proba = model13.predict_proba(X_train)[:, 1]
Y_train_pred = (Y_train_pred_proba >= 0.4).astype(int)

Y_test_pred_proba = model13.predict_proba(X_test)[:, 1]
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


model13_details = {
    'Model_ID' : 'model13',
    'Execution_time' : execution_time,
    'Model_Type' : 'DecisionTree',
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
    'DecisionTree_criterion' : 'entropy',
    'Hyperparameter_max_depth' : 10,
    'Hyperparameter_min_samples_leaf' : 8,
    'Hyperparameter_min_samples_split' : 9,
    'Boosting_type' : 'Adaboost'
}

