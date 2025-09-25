# 🥷 Projet de Détection Automatique de Fraude Bancaire – Pipeline de Production

---

## ✅ Objectif Général

Mettre en production un système de **détection automatique de fraudes bancaires**, capable de :

- 🚨 **Notifier en temps réel** lorsqu’une fraude est détectée.
- 📊 **Générer un reporting quotidien** récapitulant les paiements (frauduleux ou non) de la veille.

---

## 🧭 Étapes du Projet (dans l’ordre logique)

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

Créer 2 DAGs :

#### 🌀 `fetch_and_score` (fréquence : toutes les minutes)

- Appel de l’API → stockage brut → prédiction → insertion enrichie

#### 📆 `daily_report` (fréquence : chaque matin 7h)

- Requêtage des données de la veille
- Génération d’un rapport CSV ou HTML
- Envoi par email ou stockage dans S3

✅ Airflow devra :

- Piloter les scripts Python (`BashOperator`, `PythonOperator`)
- Ajouter des logs, gestion d’erreurs
- Être containerisé (**Docker**)

---

### 🔵 6. Reporting & Visualisation

➡️ Rendre les résultats compréhensibles pour les utilisateurs.

Deux options :

#### 📊 Dashboard interactif

- Créer un dashboard **Streamlit** ou **Hugging Face Space**
- Filtres par date
- Indicateurs clés : total transactions, % fraudes, top merchants fraudés
- Visualisations : courbes temporelles, bar charts, heatmaps

#### ✉️ Rapport automatique

- Générer un **rapport quotidien** via le DAG `daily_report`
- Format HTML ou PDF
- Envoi par **SMTP**, Zapier ou Airtable Automation

---

## 🗂️ Analyse de ton Schéma Technique (PDF fourni)

| Élément             | Présent sur ton schéma       | ✅ OK ? | Commentaires |
|---------------------|-------------------------------|--------|--------------|
| **Sources de données** | Real-time API + Historical DB | ✅     | Parfaitement identifié |
| **Modèle ML**         | scikit-learn + MLflow         | ✅     | Tu utilises XGBoost, c’est bien |
| **Orchestration**     | Airflow                       | ✅     | Bien prévu |
| **Stockage**          | Amazon S3 + NeonDB            | ✅     | Bonne séparation artefacts vs données |
| **Sortie utilisateur**| Email + Dashboard             | ✅     | Tu couvres bien les deux |
| **Déploiement UI**    | Hugging Face                  | ✅     | Excellent choix pour la démo |
| **Tracking ML**       | MLflow Tracking               | ✅     | Déjà implémenté |

✅ Ton schéma est **cohérent**, **complet**, et **opérationnel** !

---

## 🎁 Livrables Attendus

- ✅ Schéma d’**infrastructure** (déjà fait)
- 💻 **Code complet** :
  - Ingestion temps réel
  - Stockage dans DB
  - Scoring / prédiction
  - Reporting
  - Orchestration via Airflow
- 🎬 **Vidéo démo** de l’ensemble (avec Vidyard, OBS, etc.)
- 📘 *(Optionnel)* `README.md` détaillant :
  - Lancement des scripts
  - Fichiers `.env`
  - Dépendances / environnements (e.g. via `requirements.txt` ou `conda.yml`)
