# test_api_payments.py
import os
import requests
import json
from dotenv import load_dotenv
from pathlib import Path

# Charger .env
env_path = Path(__file__).parent / "secrets" / ".env"
load_dotenv(dotenv_path=env_path)

PAYMENTS_API_URL = os.getenv("PAYMENTS_API_URL")
if not PAYMENTS_API_URL:
    raise ValueError("❌ PAYMENTS_API_URL manquant dans .env")

print(f"🔗 Test API: {PAYMENTS_API_URL}")

resp = requests.get(PAYMENTS_API_URL, timeout=10)
print("📡 Status code:", resp.status_code)

# Affiche brut
print("\n--- RAW TEXT ---")
print(resp.text[:500])  # max 500 caractères

# Essai 1: .json()
try:
    data_json = resp.json()
    print("\n✅ .json() fonctionne → type:", type(data_json))
    print("Clés disponibles:", list(data_json.keys()))
except Exception as e:
    print("\n❌ .json() a échoué:", e)

# Essai 2: json.loads()
try:
    data_loads = json.loads(resp.text)
    print("\n✅ json.loads() fonctionne → type:", type(data_loads))
    print("Clés disponibles:", list(data_loads.keys()))
except Exception as e:
    print("\n❌ json.loads() a échoué:", e)
