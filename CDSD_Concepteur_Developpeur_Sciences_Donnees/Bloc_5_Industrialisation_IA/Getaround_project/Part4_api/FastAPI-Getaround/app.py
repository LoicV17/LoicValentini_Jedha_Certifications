from fastapi import FastAPI
from pydantic import BaseModel
from typing import Literal
import pandas as pd
import joblib
import os
import sys

# === Modèle attendu ===
MODEL_PATH = "model.pkl"

print("🚀 Starting API...")
print(f"🔍 Looking for model at: {MODEL_PATH}")

if not os.path.exists(MODEL_PATH):
    print(f"❌ ERROR: Model file '{MODEL_PATH}' not found.")
    sys.exit(1)

try:
    model = joblib.load(MODEL_PATH)
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"❌ ERROR while loading the model: {e}")
    sys.exit(1)

# === FastAPI app ===
app = FastAPI(
    title="🚗 Getaround Price Prediction API",
    description="Predict rental price using an XGBoost model trained on Getaround data.",
    version="1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# === Root route ===
@app.get("/")
def root():
    return {"message": "✅ API is live. Use /docs for Swagger UI."}

# === Health check route ===
@app.get("/health")
def health():
    return {"status": "API running"}

# === Input schema ===
class RentalInput(BaseModel):
    model_key: str
    mileage: int
    engine_power: int
    fuel: Literal["diesel", "petrol", "hybrid", "electric"]
    paint_color: str
    car_type: str
    private_parking_available: bool
    has_gps: bool
    has_air_conditioning: bool
    automatic_car: bool
    has_getaround_connect: bool
    has_speed_regulator: bool
    winter_tires: bool

    class Config:
        protected_namespaces = ()  # évite le warning Pydantic

# === Prediction route ===
@app.post("/predict")
def predict_price(data: RentalInput):
    try:
        input_df = pd.DataFrame([data.model_dump()])
        prediction = model.predict(input_df)
        return {"prediction": round(float(prediction[0]), 2)}
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return {"error": f"Prediction failed: {str(e)}"}
