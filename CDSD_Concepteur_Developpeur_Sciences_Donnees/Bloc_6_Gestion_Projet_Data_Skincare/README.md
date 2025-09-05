# 🩺 Bloc 6 – Deep Learning & Computer Vision

Ce dossier contient un projet réalisé dans le cadre du **Bloc 6 – Deep Learning & Computer Vision** de la certification **CDSD – Concepteur Développeur en Sciences des Données** (Jedha / RNCP 35288).  
Ce bloc vise à développer des compétences en **apprentissage profond**, **vision par ordinateur** et **industrialisation de modèles CNN**, en passant par l’entraînement, l’évaluation, l’explicabilité et la mise en production.

---

## 🧪 Projet inclus

### 🔹 SkinCare – Détection de lésions cutanées et analyse des risques
Projet complet mené sur la **classification des grains de beauté** (dataset HAM10000 et sous-datasets personnalisés).  
Objectif : détecter le caractère bénin ou malin d’une lésion cutanée et proposer une classification dermatologique en 7 classes.

---

#### 🏋️‍♂️ Entraînement des modèles
- Entraînement de plusieurs architectures CNN pré-entraînées :
  - **MobileNetV2**
  - **EfficientNetB0** (versions gelées et fine-tunées)
  - **ResNet50** (fine-tuning avancé)
- Gestion du **déséquilibre des classes** :
  - Pondération
  - Sur-échantillonnage
- Intégration de **callbacks** :
  - EarlyStopping
  - ModelCheckpoint
- Évaluation des performances :
  - Accuracy, Recall, F1-score
  - Matrices de confusion en pourcentage
  - Rapports de classification

📁 Dossier : `trained_models/`

---

#### 📊 Suivi et comparaison des performances
- Sauvegarde des résultats globaux des modèles
- Comparaison des performances entre CNN
- Calcul des risques à partir des probabilités combinées
- Génération de matrices de confusion globales avec classification en **faible / modéré / élevé**

📁 Dossier : `performance_measurement/`

---

#### 🎨 Démonstrations et visuels
- **Images de test** prêtes à l’emploi pour illustrer l’app
- Visualisation des **zones activées (Grad-CAM)** pour expliquer la prédiction du modèle
- Exemples patients pour présentation

📁 Dossier : `demo_pictures/`

---

#### 🌐 Déploiement Streamlit
- Création d’une **application interactive** Streamlit :
  - Upload ou webcam
  - Sélection d’images d’exemple
  - Entrée des métadonnées patient (âge, sexe, localisation)
  - Résultat global avec échelle de risque (**faible / modéré / élevé**)
  - Visualisation Grad-CAM
  - Top-3 prédictions multi-classes
  - Conseils dermatologiques
- Déploiement :
  - **En local** (`local_streamlit/`)
  - **Sur Hugging Face Spaces** avec requirements & runtime

📁 Dossier : `local_streamlit/`

---

#### 🖼️ Données sources
- **Dataset HAM10000** complet : 10 015 images de lésions cutanées
- Organisation en **train/val/test**
- Répertoires pour gestion des sous-datasets et scripts d’entraînement

📁 Dossier : `src_pictures/`

---

## 🧰 Technologies & outils

- **Python** (NumPy, Pandas, Scikit-learn)
- **TensorFlow / Keras** (MobileNetV2, EfficientNetB0, ResNet50)
- **Streamlit** pour l’interface utilisateur
- **Plotly** pour la visualisation
- **Grad-CAM** pour l’explicabilité
- **Hugging Face Spaces** pour le déploiement
- **Git LFS** pour le suivi des modèles et datasets volumineux

---

## 📚 Compétences mobilisées

- Prétraitement et préparation d’images médicales
- Fine-tuning et transfert learning sur CNN
- Gestion du déséquilibre des classes
- Évaluation avancée des performances (rapports, métriques médicales)
- Explicabilité (Grad-CAM)
- Développement d’une interface utilisateur interactive
- Déploiement sur Hugging Face (Streamlit)
- Organisation projet type MLOps (entraîner → mesurer → déployer)

---

👨‍🎓 Réalisé par **Loic Valentini**  
🔗 [LinkedIn](https://www.linkedin.com/in/loic-valentini-0a6238107/)  
🏫 Certification CDSD – Jedha Bootcamp  
📅 Année : 2025
