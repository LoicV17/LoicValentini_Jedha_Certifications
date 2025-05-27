import os
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import mlflow
import mlflow.sklearn

# ──────────────── Chargement des données ────────────────
df = pd.read_csv("../src/get_around_pricing_project.csv")

target_variable = "rental_price_per_day"
features_list = [col for col in df.columns if col != target_variable]

numeric_features = ['mileage', 'engine_power']
categorical_features = ['model_key', 'fuel', 'paint_color', 'car_type',
                        'private_parking_available', 'has_gps', 'has_air_conditioning',
                        'automatic_car', 'has_getaround_connect', 'has_speed_regulator', 'winter_tires']

X = df[features_list]
Y = df[target_variable]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# ──────────────── Pipeline de prétraitement + modèle ────────────────
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(drop="first", handle_unknown="ignore")

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

model = LinearRegression()

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', model)
])

# ──────────────── Initialiser MLflow ────────────────
mlflow.set_tracking_uri("mlruns")
mlflow.set_experiment("Getaround_Pricing")

with mlflow.start_run(run_name="LinearRegression"):

    # 🔁 Entraînement
    pipeline.fit(X_train, Y_train)

    # 🔍 Prédictions
    Y_test_pred = pipeline.predict(X_test)

    # 📊 Évaluation
    r2 = r2_score(Y_test, Y_test_pred)
    mae = mean_absolute_error(Y_test, Y_test_pred)
    rmse = np.sqrt(mean_squared_error(Y_test, Y_test_pred))

    # 💾 Log dans MLflow
    mlflow.log_metric("r2_score", r2)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("rmse", rmse)
    mlflow.sklearn.log_model(pipeline, "model")

    # 📁 Création du dossier de run
    run_id = mlflow.active_run().info.run_id
    run_output_dir = os.path.join("outputs", run_id)
    os.makedirs(run_output_dir, exist_ok=True)

    # 💾 Sauvegarde du modèle
    model_path = os.path.join(run_output_dir, "model.pkl")
    joblib.dump(pipeline, model_path)

    # 💾 Sauvegarde des prédictions
    preds_path = os.path.join(run_output_dir, "test_predictions.txt")
    with open(preds_path, "w") as f:
        f.write("Predictions for the test set:\n")
        f.write("\n".join([str(x) for x in Y_test_pred]))

    # 📄 Sauvegarde des métriques
    metrics_path = os.path.join(run_output_dir, "metrics.txt")
    with open(metrics_path, "w") as f:
        f.write("Evaluation Metrics:\n")
        f.write(f"R2 Score: {r2:.4f}\n")
        f.write(f"MAE: {mae:.2f}\n")
        f.write(f"RMSE: {rmse:.2f}\n")

    print(f"✅ Run terminé. Résultats sauvegardés dans: {run_output_dir}")
