---
title: SkinCare Project
emoji: 🩺
colorFrom: gray
colorTo: purple
sdk: streamlit
sdk_version: 1.44.1
app_file: app.py
pinned: false
short_description: Skincare app for Jedha certification
---

# 🔎 Skin Care – Analyse des grains de beauté

Application Streamlit permettant :
- d’estimer le **risque bénin/malin** d’une lésion cutanée à partir d’une image (modèle binaire),
- de réaliser une **classification dermatologique** en **7 classes**,
- de visualiser une **Grad-CAM** (zones d’attention du modèle),
- d’afficher une **synthèse du risque** (faible / modéré / élevé) et des **conseils**.

> ⚠️ **Avertissement** : Cette application n’est pas un dispositif médical. Elle ne remplace pas un avis clinique. En cas de doute, consultez un professionnel de santé.

---

## 🧠 Modèles & classes

- **Modèle 1** (binaire) : prédiction *bénin vs malin* → `model1_h5version.h5`  
- **Modèle 2** (multiclasse, 7 catégories) → `model2.h5`  
- **Modèle 3** (métadonnées tabulaires, stacking) → `model3.joblib`

**Classes (7)** :
- `akiec` – kératoses actiniques  
- `bcc` – carcinome basocellulaire  
- `bkl` – kératoses séborrhéiques  
- `df` – dermatofibromes  
- `mel` – mélanome  
- `nv` – nævus mélanocytaire  
- `vasc` – lésions vasculaires

---

## 📦 Structure du dépôt

├─ app.py
├─ gradcam.py
├─ requirements.txt
├─ runtime.txt
├─ classes/ # images d’exemple par classe (affichage)
│ ├─ akiec.jpg
│ ├─ ...
├─ examples/ # images d’exemple utilisateur
│ ├─ Exemple1.jpg
│ ├─ ...
├─ model1_h5version.h5 # via Git LFS
├─ model2.h5 # via Git LFS
├─ model3.joblib # via Git LFS
└─ README.md

## Dépendances et versions

python 3.10

numpy==1.26.4
pandas==2.2.2
scipy==1.13.1
tensorflow-cpu==2.17.1
keras==3.3.3
ml-dtypes==0.4.0
h5py==3.11.0
streamlit==1.37.1
Pillow==10.4.0
plotly==5.22.0
joblib==1.4.2
scikit-learn==1.5.1
protobuf==4.25.3
rich==13.7.1
packaging==24.1
pyarrow==15.0.2
matplotlib==3.8.4


## 🧑‍💻 Utilisation

Charger une image : via webcam ou upload, ou choisir un exemple.

Renseigner les métadonnées patient (facultatif) : âge, sexe, localisation.

L’application affiche :

la photo,

le résultat général (type de lésion + niveau de risque),

une jauge (probabilité malin du modèle binaire),

un Top-3 multi-classe,

la Grad-CAM (zones d’attention),

des conseils.

Si les métadonnées sont absentes, la classification s’appuie uniquement sur l’image (modèle 2).

