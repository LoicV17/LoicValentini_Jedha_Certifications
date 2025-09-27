# 🥷 Projet de détection automatique de fraude bancaire — Pipeline de prod

---

## ✅ Objectif
Mettre en production un système de **détection automatique de fraudes bancaires**.

---

## 🧭 Étapes du projet

### 🟣 1) Préparation du modèle (ML) ✅
- Récupération du dataset   
- Nettoyage : ajout des colonnes temps & distance, suppression des colonnes inutiles   
- Sauvegarde d’un dataset corrigé   
- Préprocessing pour le ML   
- Entraînement **XGBoost** avec `RandomizedSearch` pour les hyperparamètres   
- Suivi des expériences via **MLflow**   
- Sauvegarde locale du modèle `.pkl`   
- Sauvegarde des label encoders (category/state) `.pkl`   
- Export des métriques : matrice de confusion, rapport de classification, ROC et PR   
- Upload des artefacts (`mlruns`, `models/` + métriques et encoders) vers **S3** 

---

### 🟡 2) Endpoint de scoring ✅
➡️ Exposer une API pour prédire la fraude en temps réel.

- API **FastAPI** (ou Flask) chargeant le modèle depuis AWS S3  
- Endpoint `/predict` recevant une transaction JSON  
- Chargement du modèle `.pkl`  
- Tests avec plusieurs exemples réels pour valider la cohérence  
- Hébergement local  
- Déploiement sur **Hugging Face Space** → https://loicv17-fraud-detection-api.hf.space/docs

---

### 🟠 3) Ingestion temps réel & stockage ✅
➡️ Récupérer en continu les paiements et les historiser proprement.

- Script `fetch_payments.py` qui :   
  - appelle l’**API paiements** toutes les minutes   
  - stocke les transactions dans **NeonDB** (PostgreSQL cloud)   
  - applique `model.predict()` pour enrichir chaque ligne   
- Tables :  
  - `raw_payments` → transactions brutes   
  - `scored_payments` → transactions scorées 

---

## 🔵 4) Orchestration avec Airflow (Docker) ✅
Automatiser l’ETL + scoring + reporting.

### 4.1 `fetch_and_score` (toutes les minutes) ✅
Basé sur `fetch_payments.py` :
- **Ingestion** : récupère une transaction et l’insère dans `raw_payments`.  
- **Prétraitement** : features (distance client–merchant, date/heure, week-end, etc.).  
- **Scoring** : envoie la transaction à l’API ML, récupère prédiction + proba, insère dans `scored_payments`.  
- **Persistance S3** : append dans `fraud/raw_payments.csv` et `fraud/scored_payments.csv`.

---

### 4.2 `full_report_refresh` — source pour la visualisation ✅
**Fréquence : chaque heure**

- Lit `fraud/scored_payments.csv` depuis S3.  
- Convertit en Parquet optimisé → `reports/full/scored_payments.parquet`.  
- Dépose un flag `reports/full/READY`.  
- Sert de **source de vérité** pour les rapports et le dashboard Streamlit.

---

### 4.3 `daily_report_email` — rapport quotidien ✅
**Fréquence : chaque jour à 07:00**

- Lit `reports/full/scored_payments.parquet` (S3).  
- (Re)construit `event_time` si besoin, filtre les **24 dernières heures**.  
- Calcule : Transactions / Fraudes / Taux / Montant total / Montant fraudé.  
- Génère un **HTML** avec ces KPI + table détaillée des fraudes (tri par date).  
- Envoie le mail via `SmtpHook` (connexion `smtp_gmail_basic`) aux contacts `REPORT_EMAIL_TO`.  
- Archive le HTML dans S3 : `reports/daily/report_YYYYMMDD.html`.

---

### 4.4 `fraud_alert_email` — alerte temps réel ✅
**Fréquence : toutes les minutes**

- Charge le Parquet `reports/full/scored_payments.parquet` (S3) et reconstruit `event_time` si besoin.  
- Filtre les transactions **frauduleuses** (`prediction == 1`) sur une fenêtre glissante de **2 minutes** (tolérance anti-lag).  
- Prépare un email HTML (compte, montant total) + tableau des fraudes (limité à 200 lignes anti-spam).  
- Envoie via `SmtpHook` (`smtp_gmail_basic`) aux destinataires `REPORT_EMAIL_TO`.  
- N’envoie rien s’il n’y a pas de fraude ou si la liste de destinataires est vide.
