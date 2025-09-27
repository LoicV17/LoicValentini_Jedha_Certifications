# 🥷 Projet de Détection Automatique de Fraude Bancaire – Pipeline de Production

---

## ✅ Objectif Général

Mettre en production un système de **détection automatique de fraudes bancaires**

---

## 🧭 Étapes du Projet

---

### 🟣 1. Préparation du Modèle de Détection (ML) ✅ 

- Récupération du dataset ✅ 
- Nettoyage du dataset - Ajout de colonnes liées à la date et à la distance - Suppression de colonnes inutiles ✅ 
- Sauvegarde d'un dataset corrigé ✅ 
- Preprocessing pour Machine Learning ✅ 
- Entraînement du modèle **XGBoost** avec `RandomizedSearch` pour les hyperparamètres ✅ 
- Tracking et logging via **MLflow** ✅ 
- Sauvegarde locale du modèle `.pkl` ✅ 
- Sauvegarde locale des label_encoders (category/state) `.pkl` ✅ 
- Export des métriques : la matrice de confusion, le rapport de classification, le ROC et le PR ✅ 
- Upload des artefacts (`mlruns`, `models/`), contenant également métriques et encoders vers **S3** ✅  


### 🟡 2. Création d’un Endpoint de Scoring ✅ 

➡️ Permettre un appel direct via API pour prédire une fraude à la volée.

- Créer une API avec **FastAPI** (ou Flask) utilisant le modèle stocké sur AWS S3 ✅ 
- Endpoint `/predict` prenant une transaction JSON en entrée  ✅ 
- Chargement du modèle depuis `.pkl` ✅ 
- Tester avec plusieurs exemples réels JSON pour vérifier la cohérence des résultats ✅ 
- Héberger l'API en local ✅ 
- Héberger sur **Hugging Face Space** ✅ -> https://loicv17-fraud-detection-api.hf.space/docs

---

### 🟠 3. Ingestion Temps Réel des Transactions et Stockage des données ✅ 

➡️ Récupérer en continu les données de paiement depuis l’API externe.
➡️ Organiser les données de manière durable pour les analyses futures.

- Créer un script `fetch_payments.py` qui : ✅ 
  -> Appelle l’**API des paiements** toutes les minutes ✅ 
  -> Stocke les transactions dans **NeonDB** (PostgreSQL cloud) ✅ 
  -> Applique `model.predict()` pour enrichir chaque ligne ✅ 

  -> Table `raw_payments` → transactions brutes ✅ 
  -> Table `scored_payments` → transactions enrichies de la prédiction ✅ 

---

### 🔵 4. Orchestration avec Airflow

➡️ Automatiser l’ensemble du pipeline ETL + scoring + reporting.

Installer un Airflow contenerisé (**Docker**) ✅



### 🔵 4.1 DAG fetch_and_score ✅

#### 🌀 `fetch_and_score` (fréquence : toutes les minutes), basé sur `fetch-payments.py` ✅

- Ingestion : appelle l'API de paiements pour récupérer une transaction brute et l’insère dans la table raw_payments (NeonDB/Postgres).
- Prétraitement : enrichit la transaction avec des features (distance client–merchant, date/heure décomposée, indicateurs week-end, etc.).
- Scoring : envoie la transaction transformée à une API de scoring ML, récupère la prédiction + probabilité, et insère le résultat dans la table scored_payments.
- Persistance S3 : ajoute la transaction brute et la transaction scorée dans des fichiers CSV versionnés sur S3 (fraud/raw_payments.csv et fraud/scored_payments.csv).



### 🔵 4.2 DAG full_report_refresh -> Visualisation permanente ✅

#### 📆 `full_report_refresh` (fréquence : chaque heure) ✅

- Chargement : lit le CSV brut fraud/scored_payments.csv depuis S3.
- Transformation : convertit ce CSV en Parquet optimisé (reports/full/scored_payments.parquet) pour faciliter la consommation par Streamlit.
- Signalisation : crée un fichier READY dans reports/full/READY pour indiquer que les données sont prêtes.
- Utilité : alimente et synchronise la source de vérité pour les rapports et le dashboard Streamlit.



### 🔵 4.3 DAG daily_report_email ✅

#### 📧 `daily_report_email` (fréquence : chaque jour à 7h) ✅

- Chargement des données : lit le parquet reports/full/scored_payments.parquet depuis S3 (bucket pris dans AIRFLOW_S3_BUCKET).
- Fenêtre temporelle : (re)construit event_time si nécessaire, puis filtre les transactions des 24 dernières heures.
- Calculs & rendu : calcule Transactions / Fraudes / Taux (%) / Montant total / Montant fraudé, génère un HTML avec ces KPI + table détaillée des fraudes (triées par date).
- Envoi d’email : expédie le rapport HTML via SmtpHook et la connexion smtp_gmail_basic aux destinataires listés dans REPORT_EMAIL_TO.
- Archivage : enregistre le même HTML dans S3 sous reports/daily/report_YYYYMMDD.html



### 🔵  4.4 DAG fraud_alert_email ✅

#### 🚨 `fraud_alert_email` (fréquence : toutes les minutes) ✅
- Charge le parquet reports/full/scored_payments.parquet depuis S3 et reconstruit event_time si besoin à partir des colonnes trans_*.
- Filtre les transactions frauduleuses (prediction==1) sur la fenêtre glissante des 2 dernières minutes (tolérance anti-lag).
- Construit un e-mail HTML avec un récap (compte, montant total) et un tableau des fraudes (limité à 200 lignes anti-spam).
- Envoie l’alerte via SmtpHook (connexion smtp_gmail_basic) aux destinataires REPORT_EMAIL_TO – et n’envoie rien s’il n’y a pas de fraude ou pas de destinataires.






## 🎁 Livrables Attendus

- ✅ Schéma d’**infrastructure** (déjà fait)
- 🎬 **Vidéo démo** de l’ensemble (avec Vidyard, OBS, etc.)
- 📘 *(Optionnel)* `README.md` détaillant :
  - Lancement des scripts
  - Fichiers `.env`
  - Dépendances / environnements (e.g. via `requirements.txt` ou `conda.yml`)
