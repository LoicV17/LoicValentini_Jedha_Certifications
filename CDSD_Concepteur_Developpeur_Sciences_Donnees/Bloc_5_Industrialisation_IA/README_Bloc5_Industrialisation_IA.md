# 🚗 Bloc 5 – Industrialisation de l’IA

Ce dossier contient un projet réalisé dans le cadre du **Bloc 5 – Industrialisation de l’IA** de la certification **CDSD – Concepteur Développeur en Sciences des Données** (Jedha / RNCP 35288).  
Ce bloc vise à développer des compétences en **mise en production de modèles de machine learning**, en passant par le suivi d’expérimentations, le déploiement d’API, et la construction d’interfaces utilisateur.

---

## 🧪 Projet inclus

### 🔹 Getaround – Optimisation des retards et des prix de location
Projet complet mené pour la plateforme de location de voitures **Getaround**, comportant deux volets principaux :

#### 📈 Analyse des retards de location
- Étude des marges de sécurité entre réservations
- Évaluation de l’impact d’un buffer sur :
  - Les retards
  - Les cas critiques évités
  - Les annulations concernées
- Création d’un **dashboard interactif Streamlit** permettant :
  - Visualisation de KPIs
  - Simulation de seuils de sécurité
  - Filtres par type de check-in

📁 Dossiers : `Part1_eda/` et `Part2_streamlit/`

#### 🤖 Prédiction du prix de location
- Nettoyage et préparation des données issues de Getaround (`get_around_pricing_project.csv`)
- Entraînement d’un **modèle de régression (XGBoost)** avec preprocessing (scaling + one-hot encoding)
- Suivi des expérimentations avec **MLflow en local** :
  - Paramètres
  - Performances
  - Artéfacts du modèle
- Sauvegarde du modèle dans un fichier `.pkl` pour intégration en production

📁 Dossier : `Part3_machine_learning/`

#### 📚 Création et tracking des modèle Machine learning
- Tests de 3 modèles de prédiction du prix de location en fonction des caractéristiques fournies **LinearReg, RandomForeest, XGBoost**
- Tracking des modèles avec **MLFlow** pour détecter le modèle le plus performant
- Hébergement de MLFlow sur **Hugging Face Spaces**
- Sauvegarde du modèle en `.pkl`


📁 Dossier : `Part4_api/`
#### 🧩 Déploiement de l’API de prédiction
- Développement d’une **API FastAPI** avec documentation interactive (`/docs`)
- Création de l’endpoint `/predict` pour effectuer une prédiction de prix à partir des caractéristiques d’un véhicule
- Hébergement de l’API sur **Hugging Face Spaces**
- Intégration du modèle `.pkl` et validation des entrées via Pydantic
- Test de l’API en local et en production
---

## 🧰 Technologies & outils
n
- Python (Pandas, Scikit-learn, XGBoost)
- MLflow pour le suivi des modèles
- Streamlit pour la visualisation
- FastAPI pour l’API de prédiction
- Hugging Face Spaces pour le déploiement
- Git / GitHub pour le versioning
- Docker (en option pour la conteneurisation locale)

---

## 📚 Compétences mobilisées

- Préparation de données et feature engineering
- Modélisation et évaluation de modèles ML
- Suivi des expériences (MLOps)
- Développement et documentation d’API RESTful
- Industrialisation et déploiement de projets IA
- Visualisation interactive orientée produit

---

👨‍🎓 Réalisé par **Loic Valentini**  
🔗 [LinkedIn](https://www.linkedin.com/in/loic-valentini-0a6238107/)  
🏫 Certification CDSD – Jedha Bootcamp  
📅 Année : 2025
