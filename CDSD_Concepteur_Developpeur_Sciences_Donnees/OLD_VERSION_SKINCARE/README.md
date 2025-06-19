---
title: SkinCare Project
emoji: 🩺
colorFrom: gray
colorTo: purple
sdk: streamlit
sdk_version: 1.44.1
app_file: app.py
pinned: false
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

---

# 🔎 Skin Care - Analyse des Grains de Beauté

L'application **Skin Care** permet de prédire le caractère bénin ou malin d'un grain de beauté à partir d'une image et d'effectuer une classification dermatologique sur la base de 7 types de lésions cutanées. Elle utilise deux modèles pré-entrainés : l'un pour la classification bénin/malin, et l'autre pour la classification détaillée des lésions cutanées.

## Fonctionnalités

- **Prédiction bénin/malin** : Le modèle prédit la probabilité qu'un grain de beauté soit bénin ou malin.
- **Classification dermatologique** : Le modèle classe l'image dans l'une des sept catégories dermatologiques : 
  - `akiec` (kératoses actiniques)
  - `bcc` (carcinome basocellulaire)
  - `bkl` (kératoses séborrhéiques)
  - `df` (dermatofibromes)
  - `mel` (mélanome)
  - `nv` (névus mélanocytaire)
  - `vasc` (lésions vasculaires)
- **Visualisation Grad-CAM** : Génération d'une carte de chaleur Grad-CAM pour visualiser les zones de l'image ayant impacté la décision du modèle.
- **Analyse du risque** : Détection du risque (élevé, modéré, faible) sur la base des prédictions.
- **Conseils Skincare** : Fourniture de conseils en fonction du niveau de risque détecté.

## Technologies

- **Streamlit** : Interface interactive permettant à l'utilisateur de charger une image et de visualiser les résultats.
- **TensorFlow** : Utilisation de modèles Keras pour les prédictions.
- **Matplotlib** : Génération de visualisations telles que la jauge de probabilité.
- **Grad-CAM** : Visualisation des zones de l'image qui ont contribué à la prédiction.
- **Pillow** : Traitement d'images.

## Prérequis

1. **Python 3.x** installé.
2. Bibliothèques suivantes :
   - `streamlit`
   - `tensorflow`
   - `numpy`
   - `matplotlib`
   - `Pillow`
   - `os`

Vous pouvez installer ces bibliothèques via `pip` :

```bash
pip install streamlit tensorflow numpy matplotlib Pillow

