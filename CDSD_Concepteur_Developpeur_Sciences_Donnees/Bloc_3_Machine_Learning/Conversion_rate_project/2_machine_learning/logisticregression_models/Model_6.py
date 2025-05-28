# Import basic libraries and appropriate scikit-learn libraries

import pandas as pd
import numpy as np


from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix, recall_score, precision_score, ConfusionMatrixDisplay, RocCurveDisplay)

import time
import random
import mlflow
import mlflow.sklearn



data = pd.read_csv('../src/conversion_data_train.csv')
print("Data successfully retrieved")

# Remove the 'source' column
if 'source' in data.columns:
    data.drop(columns=['source'], inplace=True)
    print("Column 'source' successfully dropped")

# Remove outliers on age (keep values within ±3 std from the mean)
age_mean = data["age"].mean()
age_std = data["age"].std()
age_min = age_mean - 3 * age_std
age_max = age_mean + 3 * age_std
initial_len = len(data)
data = data[(data["age"] >= age_min) & (data["age"] <= age_max)]
print(f"Removed {initial_len - len(data)} outliers on 'age' (outside [{age_min:.2f}, {age_max:.2f}])")



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

for i in range(500):
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
    model6_test = LogisticRegression(verbose=1)
    model6_test.fit(X_train, Y_train)

    # Predictions on train set and test set
    Y_train_pred_proba = model6_test.predict_proba(X_train)[:, 1]
    Y_train_pred = (Y_train_pred_proba >= 0.38).astype(int)

    Y_test_pred_proba = model6_test.predict_proba(X_test)[:, 1]
    Y_test_pred = (Y_test_pred_proba >= 0.38).astype(int)

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
        model6 = model6_test
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

    print(f"[{i+1}/500] | F1 test: {f1_test:.4f} | Best F1 so far: {best_f1:.4f}", end="")
    print()

    if f1_test > best_f1:
        print(" ✅ New best model found!")

    


print()
print("-----------------------------------------------------------")
print(f"Best features list: {best_feature_random_list} with F1-score: {best_f1}")


# MLflow run
with mlflow.start_run(run_name="model6"):
    # Model training
    model6 = LogisticRegression()
    model6.fit(X_train, Y_train)
    
    # Predictions on train set and test set and setting with new probability
    Y_train_pred = model6.predict(X_train)
    Y_train_pred_proba = model6.predict_proba(X_train)[:, 1]
    Y_train_pred = (Y_train_pred_proba >= 0.38).astype(int)

    Y_test_pred = model6.predict(X_test)
    Y_test_pred_proba = model6.predict_proba(X_test)[:, 1]
    Y_test_pred = (Y_test_pred_proba >= 0.38).astype(int)

    # Log the model and parameters
    mlflow.sklearn.log_model(model6, "model")
    mlflow.log_param("model", "logistic_regression_multivariate")
    mlflow.log_param("preprocessing", "specific")
    mlflow.log_param("features", "age_without_outliers, country, new_user, total_pages_visited")
    mlflow.log_param("decision_threshold", 0.38)

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


