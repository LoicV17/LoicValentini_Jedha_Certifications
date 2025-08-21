# Skin Disease Classification - MobileNetV2 Fine-tuned

Ce modèle est une version améliorée du modèle MobileNetV2 précédemment entraîné sans fine-tuning.

## 📦 Architecture

- **Backbone** : MobileNetV2 (pré-entraîné ImageNet)
- **Fine-tuning** : oui, dégel des **20 dernières couches**
- **Couche de sortie** : Dense(1, activation='sigmoid') pour classification binaire

## 📁 Données utilisées

- 📂 `train_skindisease/` pour l'entraînement
- 📂 `val_skindisease/` pour la validation
- 📂 `test_skindisease/` pour l'évaluation finale
- Deux classes : `benign`, `malignant` (dataset équilibré)

## ⚙️ Entraînement

- **Taille d’image** : 224 x 224
- **Époques** : 10
- **Batch size** : 32
- **Optimiseur** : Adam (lr=1e-4)
- **Callbacks** :
  - `ModelCheckpoint` (sauvegarde du meilleur modèle `.h5`)
  - `EarlyStopping` (patience=3, restore_best_weights)

## 🎯 Résultats

- **train_accuracy final** : ~99.8%
- **val_accuracy max** : ~83%
- Le modèle a convergé rapidement, mais l'écart train/val suggère un **début d'overfitting** après l’époque 4.

## 📈 Sorties générées

- `model2_best_model.h5` : modèle sauvegardé
- `accuracy_curve_finetuned.png` et `loss_curve_finetuned.png` : courbes d’apprentissage
- `classification_report_finetuned.txt` : rapport de précision/rappel/F1-score
- `confusion_matrix_finetuned.png` : matrice de confusion sur `test_skindisease/`

## Lancement

```bash
python train_model2.py --epochs 10 --batch_size 32
