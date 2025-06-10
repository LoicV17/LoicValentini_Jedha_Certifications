import requests

# URL /predict
url = "https://loicv17-fastapi-getaround.hf.space/predict"

# Exemple de données à envoyer
data = {
    "model_key": "Peugeot",
    "mileage": 80000,
    "engine_power": 110,
    "fuel": "diesel",
    "paint_color": "black",
    "car_type": "convertible",
    "private_parking_available": True,
    "has_gps": True,
    "has_air_conditioning": True,
    "automatic_car": False,
    "has_getaround_connect": True,
    "has_speed_regulator": True,
    "winter_tires": True
}

# Envoi de la requête POST
response = requests.post(url, json=data)

# Affichage du résultat
if response.status_code == 200:
    print("✅ Prediciton :", response.json())
else:
    print("❌ Erreur :", response.status_code, response.text)
