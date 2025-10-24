# ==================================================
# Stripe Data Project — Étape 5 : NoSQL Analytics (MongoDB)
# Author : Loïc Valentini
# Date   : 2025-10-25
# Purpose: Exécuter les requêtes d'analyse MongoDB via PyMongo
# ==================================================

from pymongo import MongoClient
from datetime import datetime, timedelta

# ==================================================
# 1) Connexion à MongoDB
# (adapter URI selon environnement)
# ==================================================
client = MongoClient("mongodb://localhost:27017/")
db = client["stripe_nosql"]  # nom de la base NoSQL

print("✅ Connected to MongoDB")

# ==================================================
# 2️⃣ Requêtes d'analyse
# ==================================================

# 1️⃣ Taux d’échec par type d’événement (24h)
pipeline_failure_rate = [
    {"$match": {"ts": {"$gte": datetime.utcnow() - timedelta(days=1)}}},
    {"$group": {
        "_id": "$event",
        "total": {"$sum": 1},
        "failed": {"$sum": {"$cond": [{"$eq": ["$status", "failed"]}, 1, 0]}}
    }},
    {"$project": {
        "_id": 0,
        "event": "$_id",
        "failure_rate": {"$divide": ["$failed", "$total"]}
    }},
    {"$sort": {"failure_rate": -1}}
]
res1 = list(db.clickstream_events.aggregate(pipeline_failure_rate))
print("\n1️⃣ Taux d’échec par type d’événement (24h) :")
for r in res1:
    print(r)

# 2️⃣ Temps moyen de session par pays
pipeline_avg_session = [
    {"$group": {"_id": "$geo.country", "avg_duration": {"$avg": "$duration_sec"}}},
    {"$sort": {"avg_duration": -1}}
]
res2 = list(db.sessions.aggregate(pipeline_avg_session))
print("\n2️⃣ Temps moyen de session par pays :")
for r in res2[:10]:
    print(r)

# 3️⃣ Clients avec comportements suspects (trop d’échecs)
res3 = list(db.fraud_features.find(
    {"metrics.tx_failed": {"$gte": 10}},
    {"_id": 0, "customer_id": 1, "metrics.tx_failed": 1}
).sort("metrics.tx_failed", -1))
print("\n3️⃣ Clients suspects (>=10 échecs) :")
for r in res3[:10]:
    print(r)

# 4️⃣ Moyenne des scores de fraude par modèle
pipeline_model_scores = [
    {"$group": {
        "_id": "$model.name",
        "avg_score": {"$avg": "$score"},
        "count": {"$sum": 1}
    }},
    {"$sort": {"avg_score": -1}}
]
res4 = list(db.model_scores.aggregate(pipeline_model_scores))
print("\n4️⃣ Moyenne des scores de fraude par modèle :")
for r in res4:
    print(r)

# 5️⃣ Top IP par volume d’échecs (1h)
pipeline_top_ip = [
    {"$match": {"status": "failed", "ts": {"$gte": datetime.utcnow() - timedelta(hours=1)}}},
    {"$group": {"_id": "$geo.ip", "fails": {"$sum": 1}}},
    {"$sort": {"fails": -1}},
    {"$limit": 10}
]
res5 = list(db.clickstream_events.aggregate(pipeline_top_ip))
print("\n5️⃣ Top IP par volume d’échecs (1h) :")
for r in res5:
    print(r)

# ==================================================
# 6️⃣ Résumé des analyses
# ==================================================
print("\n✅ Analyses terminées.")
print("Collections consultées : clickstream_events, sessions, fraud_features, model_scores")
