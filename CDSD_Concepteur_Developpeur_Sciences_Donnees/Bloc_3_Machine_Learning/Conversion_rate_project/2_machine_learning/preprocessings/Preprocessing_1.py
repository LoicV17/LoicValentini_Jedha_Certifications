# Import basic libraries and appropriate scikit-learn libraries

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer


data = pd.read_csv('../src/conversion_data_train.csv')
print("Data successfully retrieved")

# Multivariate model

features_list = ['country','age','new_user','source','total_pages_visited']
numeric_features = ['age','total_pages_visited']
categorical_features = ['country', 'source', 'new_user']
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

# Preprocessing

numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(drop="first")

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)
print()
print("Preprocessing successfully defined")

# Fit transform on train data
X_train = preprocessor.fit_transform(X_train)
print()
print("Training data successfully standardized")

# Transform only on test data
X_test = preprocessor.transform(X_test)
print()
print("Test data successfully standardized")


# Label_encoding
encoder = LabelEncoder()
Y_train = encoder.fit_transform(Y_train)
Y_test = encoder.transform(Y_test)
print()
print("Encoding successfully performed")



