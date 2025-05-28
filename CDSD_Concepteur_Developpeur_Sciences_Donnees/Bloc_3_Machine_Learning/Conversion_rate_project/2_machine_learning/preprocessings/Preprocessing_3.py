import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer

# Load data
data = pd.read_csv('../src/conversion_data_train.csv')
print("Data successfully retrieved")

# ✅ Remove outliers on age (mean ± 3*std)
age_mean = data["age"].mean()
age_std = data["age"].std()
age_min = age_mean - 3 * age_std
age_max = age_mean + 3 * age_std
data = data[(data["age"] >= age_min) & (data["age"] <= age_max)]
print(f"Data filtered for age outliers (range: {age_min:.2f} to {age_max:.2f})")

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

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0, stratify=Y)
print()
print("Train-test split successfully done")

# Preprocessing pipeline
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

# Fit-transform on training set
X_train = preprocessor.fit_transform(X_train)
print()
print("Training data successfully standardized")

# Transform test set
X_test = preprocessor.transform(X_test)
print()
print("Test data successfully standardized")

# Encode labels
encoder = LabelEncoder()
Y_train = encoder.fit_transform(Y_train)
Y_test = encoder.transform(Y_test)
print()
print("Encoding successfully performed")
