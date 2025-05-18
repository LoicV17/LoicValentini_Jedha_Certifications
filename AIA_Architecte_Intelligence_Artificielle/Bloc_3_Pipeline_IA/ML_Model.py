# 📦 Imports
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

import xgboost as xgb

import os
from pathlib import Path

# 📥 Charger les données
url = "https://lead-program-assets.s3.eu-west-3.amazonaws.com/M05-Projects/fraudTest.csv"
df = pd.read_csv(url)

# 🧼 Nettoyage initial
df.drop(columns=["Unnamed: 0"], inplace=True)
df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
df["gender"] = df["gender"].map({"F": 0, "M": 1})
df["distance"] = ((df["lat"] - df["merch_lat"])**2 + (df["long"] - df["merch_long"])**2) ** 0.5
df["merchant"] = df["merchant"].str.replace(r'^fraud_', '', regex=True)

# 📊 Feature engineering
df["hour"] = df["trans_date_trans_time"].dt.hour
df["dayofweek"] = df["trans_date_trans_time"].dt.dayofweek

# 🔤 Encodage de la variable catégorielle
le = LabelEncoder()
df["category"] = le.fit_transform(df["category"])

# 🎯 Cible + Features
drop_cols = [
    "cc_num", "first", "last", "street", "city", "state", "zip",
    "job", "dob", "trans_num", "unix_time", "lat", "long",
    "merch_lat", "merch_long", "merchant", "trans_date_trans_time"
]
df_model = df.drop(columns=drop_cols)

X = df_model.drop(columns=["is_fraud"])
y = df_model["is_fraud"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 🎛️ Paramètres pour GridSearch
param_grid = {
    "n_estimators": [50, 100],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.1, 0.2],
    "scale_pos_weight": [1, 10],
}

# ⚙️ Modèle XGBoost
xgb_clf = xgb.XGBClassifier(eval_metric="logloss", random_state=42)

# 🔍 GridSearchCV
grid_search = GridSearchCV(
    estimator=xgb_clf,
    param_grid=param_grid,
    cv=3,
    scoring="f1",
    verbose=1,
    n_jobs=-1
)

# 🚀 Lancer un run MLflow

mlruns_path = Path("AIA_Architecte_Intelligence_Artificielle/Bloc_3_Pipeline_IA/mlruns").resolve().as_uri()
mlflow.set_tracking_uri(mlruns_path)

with mlflow.start_run(run_name="XGBoost_Fraud_GridSearch"):

    grid_search.fit(X_train, y_train)

    # 📈 Meilleurs paramètres
    best_params = grid_search.best_params_
    f1_train = round(grid_search.best_score_, 4)
    y_pred = grid_search.predict(X_test)

    # 🎯 Métriques
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    # 📝 Log des hyperparamètres
    mlflow.log_params(best_params)

    # 📊 Log des métriques
    mlflow.log_metric("f1_train_cv", f1_train)
    mlflow.log_metric("precision", report["1"]["precision"])
    mlflow.log_metric("recall", report["1"]["recall"])
    mlflow.log_metric("f1_test", report["1"]["f1-score"])

    # 💾 Log du modèle
    mlflow.sklearn.log_model(grid_search.best_estimator_, "model")

    # ✅ Affichage résumé
    print("✅ Meilleurs hyperparamètres :", best_params)
    print("✅ Score F1 moyen (train, CV) :", f1_train)
    print("\n📈 Rapport de classification sur test :")
    print(classification_report(y_test, y_pred))

    # 📉 Matrice de confusion
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Non fraude", "Fraude"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Matrice de confusion - XGBoost (Test set)")
    plt.show()
