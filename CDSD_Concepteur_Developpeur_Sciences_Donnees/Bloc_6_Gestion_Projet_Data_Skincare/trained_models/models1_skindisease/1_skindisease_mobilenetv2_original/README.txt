# Skin Disease Classification - MobileNetV2

Ce projet utilise le modèle MobileNetV2 pré-entraîné sur ImageNet pour réaliser une classification binaire entre deux types de lésions cutanées : benign et malignant.

## Choix techniques

- 📦 **Backbone** : MobileNetV2 (`include_top=False`, gelé)
- 🧠 **Fine-tuning** : non activé (les poids du backbone restent gelés)
- 🎯 **Tâche** : classification binaire (sigmoid)
- 🗂️ **Chargement des données** : image_dataset_from_directory sur `train_skindisease/`, `val_skindisease/`, `test_skindisease/`
- 📐 **Taille d’image** : 224 x 224 (dimension native de MobileNetV2)
- 🧪 **Prétraitement** : `preprocess_input()` MobileNetV2 (normalisation entre -1 et 1)
- ⚙️ **Optimiseur** : Adam
- 📉 **Fonction de perte** : binary_crossentropy
- 📊 **Metrics** : accuracy
- 🛑 **Callbacks** : 
  - `EarlyStopping` (patience = 3, restore_best_weights)
  - `ModelCheckpoint` (sauvegarde du meilleur modèle `.h5`)
- 📈 **Visualisation** : courbes loss / accuracy sauvegardées au format `.png`
- 📋 **Évaluation finale** : `classification_report` + matrice de confusion sur `test_skindisease`

## Résultats

- Meilleure **val_accuracy** observée : ~83.3%
- Durée d'entraînement : ~5 minutes

## Lancement

```bash
python train_model1.py --epochs 10 --batch_size 32
