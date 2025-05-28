# Import basic libraries and appropriate scikit-learn libraries

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix, recall_score, precision_score, ConfusionMatrixDisplay, RocCurveDisplay)

import time
import random

data = pd.read_csv('conversion_data_train.csv')
print("Data successfully retrieved")

# To define the start time
start_time = time.time()

# Mappings

region_mapping = {
    'China': 'Asia',
    'UK': 'Europe',
    'Germany': 'Europe',
    'US': 'North America'
}

tpv_bins = [0, 9, 12, 16, 30]
tpv_labels = ['very low', 'low', 'medium', 'high']

age_bins = [0, 20, 35, 100]
age_labels = ['young', 'medium', 'old']

def feature_engineering(data, feature_random_list):
    
    # All the possible features engineering possible

    if feature_random_list[0]: data['age²'] = data['age'] ** 2
    if feature_random_list[1]: data['age³'] = data['age'] ** 3
    if feature_random_list[2]: data['1/age'] = 1 / (data['age'])
    if feature_random_list[3]: data['tpv²'] = data['total_pages_visited'] ** 2
    if feature_random_list[4]: data['tpv³'] = data['total_pages_visited'] ** 3
    if feature_random_list[5]: data['1/tpv'] = 1 / (data['total_pages_visited'])
    if feature_random_list[6]: data['a*tpv'] = data['age'] * data['total_pages_visited']
    if feature_random_list[7]: data['a/tpv'] = data['age'] / (data['total_pages_visited'])
    if feature_random_list[8]: data['tpv/age'] = data['total_pages_visited'] / (data['age'])
    if feature_random_list[9]: data['region'] = data['country'].map(region_mapping)
    if feature_random_list[10]: data['visit_category'] = pd.cut(data['total_pages_visited'], bins=tpv_bins, labels=tpv_labels, include_lowest=True)
    if feature_random_list[11]: data['age_category'] = pd.cut(data['age'], bins=age_bins, labels=age_labels, include_lowest=True)

    return(data)

# Variable initialization

best_f1 = 0
best_feature_random_list = [0]*12
model_3 = None
best_numeric_features = []
best_categorical_features = []
best_f1_train = 0
best_f1_test = 0
best_accuracy_train = 0
best_accuracy_test = 0
best_precision_train = 0
best_precision_test = 0
best_recall_train = 0
best_recall_test = 0


# Loop creation

random.seed(42) # Random_state defined

for _ in range(50):
    feature_random_list = [random.choice([0, 1]) for _ in range(12)]

    data_engineered = data.copy()
    data_engineered = feature_engineering(data_engineered, feature_random_list) # Application of random feature engineering

    X = data_engineered.drop(columns=['converted'])
    Y = data_engineered['converted']

    features_list = X.columns

    # Diving features between numeric and categorical

    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()

    # Moving new_user to the good feature category
    numeric_features = [col for col in numeric_features if col != "new_user"]
    categorical_features.append("new_user")


    # Dividing into train_set and test_set 
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0, stratify=Y)

    # Preprocessing

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(drop="first")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    # Fit transform on train data
    X_train = preprocessor.fit_transform(X_train)

    # Transform only on test data
    X_test = preprocessor.transform(X_test)

    # Label_encoding
    encoder = LabelEncoder()
    Y_train = encoder.fit_transform(Y_train)
    Y_test = encoder.transform(Y_test)

    # Logistic regression model to train
    model10_test = RandomForestClassifier(n_estimators=68, max_depth=10, min_samples_leaf=5, min_samples_split=5)
    model10_test.fit(X_train, Y_train)

    # Predictions on train set and test set
    Y_train_pred_proba = model10_test.predict_proba(X_train)[:, 1]
    Y_train_pred = (Y_train_pred_proba >= 0.4).astype(int)

    Y_test_pred_proba = model10_test.predict_proba(X_test)[:, 1]
    Y_test_pred = (Y_test_pred_proba >= 0.4).astype(int)

    f1_train = f1_score(Y_train,Y_train_pred)
    f1_test = f1_score(Y_test, Y_test_pred)
    accuracy_train = accuracy_score(Y_train,Y_train_pred)
    accuracy_test = accuracy_score(Y_test, Y_test_pred)
    precision_train = precision_score(Y_train,Y_train_pred)
    precision_test = precision_score(Y_test, Y_test_pred)
    recall_train = recall_score(Y_train,Y_train_pred)
    recall_test = recall_score(Y_test, Y_test_pred)

    if f1_test > best_f1:
        best_f1 = f1_test
        best_feature_random_list = feature_random_list
        model10 = model10_test
        best_numeric_features = numeric_features
        best_categorical_features = categorical_features
        best_f1_train = f1_train
        best_f1_test = f1_test
        best_accuracy_train = accuracy_train
        best_accuracy_test = accuracy_test
        best_precision_train = precision_train
        best_precision_test = precision_test
        best_recall_train = recall_train
        best_recall_test = recall_test



print()
print("-----------------------------------------------------------")
print(f"Best features list: {best_feature_random_list} with F1-score: {best_f1}")


# To define the end time
end_time = time.time()

# To define the execution time
execution_time = end_time - start_time


model10_details = {
    'Model_ID' : 'model10',
    'Execution_time' : execution_time,
    'Model_Type' : 'RandomForest',
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
    'Hyperparameter_min_samples_leaf' : 5,
    'Hyperparameter_min_samples_split' : 5,
    'Hyperparameter_N_estimators' : 68
}



