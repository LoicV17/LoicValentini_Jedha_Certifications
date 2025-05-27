import requests
import json

url = "https://loicv17-api-getaround.hf.space/predict"  # Remplacez par l'URL de ton API de prédiction
data = {
    "model_key" : "Citroën",
    "mileage": 140411,
    "engine_power": 100,
    "fuel": "diesel",
    "paint_color": "black",
    "car_type": "convertible",
    "private_parking_available": True,
    "has_gps": True,
    "has_air_conditioning": False,
    "automatic_car": False,
    "has_getaround_connect": True,
    "has_speed_regulator": True,
    "winter_tires": True
}

headers = {"Content-Type": "application/json"}

# Envoi de la requête POST à l'API
response = requests.post(url, data=json.dumps(data), headers=headers)

if response.status_code == 200:
    print("API Response:", response.json())
else:
    print("API Error:", response.status_code, response.text)
