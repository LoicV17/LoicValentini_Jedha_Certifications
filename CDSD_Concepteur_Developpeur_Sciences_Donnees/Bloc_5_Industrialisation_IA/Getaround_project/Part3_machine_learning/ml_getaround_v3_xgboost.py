import os
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from xgboost import XGBRegressor
import mlflow
import mlflow.sklearn

# ───── Load data ─────
df = pd.read_csv("../src/get_around_pricing_project.csv")

target_variable = "rental_price_per_day"
features = [col for col in df.columns if col != target_variable]

numeric_features = ['mileage', 'engine_power']
categorical_features = ['model_key', 'fuel', 'paint_color', 'car_type',
                        'private_parking_available', 'has_gps', 'has_air_conditioning',
                        'automatic_car', 'has_getaround_connect', 'has_speed_regulator', 'winter_tires']

X = df[features]
y = df[target_variable]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# ───── Preprocessing ─────
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(drop="first", handle_unknown="ignore")

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# ───── XGBoost Model ─────
model = XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=0)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", model)
])

# ───── MLflow Tracking ─────
mlflow.set_tracking_uri("mlruns")
mlflow.set_experiment("Getaround_Pricing")

with mlflow.start_run(run_name="XGBoost"):

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    mlflow.log_param("model", "XGBoost")
    mlflow.log_param("n_estimators", 200)
    mlflow.log_param("max_depth", 6)
    mlflow.log_param("learning_rate", 0.1)

    mlflow.log_metric("r2_score", r2)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("rmse", rmse)
    mlflow.sklearn.log_model(pipeline, "model")

    run_id = mlflow.active_run().info.run_id
    output_dir = os.path.join("outputs", run_id)
    os.makedirs(output_dir, exist_ok=True)

    joblib.dump(pipeline, os.path.join(output_dir, "model.pkl"))

    with open(os.path.join(output_dir, "test_predictions.txt"), "w") as f:
        f.write("Predictions:\n")
        f.write("\n".join(map(str, y_pred)))

    with open(os.path.join(output_dir, "metrics.txt"), "w") as f:
        f.write("Evaluation Metrics:\n")
        f.write(f"R2 Score: {r2:.4f}\n")
        f.write(f"MAE: {mae:.2f}\n")
        f.write(f"RMSE: {rmse:.2f}\n")

    print(f"✅ XGBoost run logged under outputs/{run_id}")
