# Modèle 4 : EfficientNetB0 Fine-Tuné pour la Classification de Lésions Cutanées

## 📁 Dossier
`4_skindisease_efficientnetb0_finetuned`

## 🧠 Architecture
- Base : `EfficientNetB0` (ImageNet weights)
- Fine-tuning : 40 dernières couches dégelées
- Couches ajoutées : GlobalAveragePooling + Dense(1, sigmoid)
- Freeze initial partiel pour profiter du transfert d’apprentissage tout en s’adaptant aux spécificités de notre dataset

## ⚙️ Paramètres d'entraînement
- Batch size : 16
- Epochs : 40 (early stopping après 8)
- Optimizer : Adam (learning rate par défaut)
- Perte : Binary Crossentropy
- Métriques : Accuracy, AUC
- EarlyStopping sur val_loss avec patience = 3

## 🧪 Données
- Dataset : `train_skindisease`, `val_skindisease` (images 224x224, 2 classes)
- Split : 60% train / 20% val / 20% test (non touché)

## 🧾 Résultats
- Meilleur modèle sauvegardé : `model4_best_model.h5`
- Meilleure val_accuracy : **~87%**
- Meilleur val_auc : **~0.95**
- Courbes sauvegardées :
  - `accuracy_curve.png`
  - `loss_curve.png`
- Rapport de classification + matrice de confusion générés sur `test_skindisease`

## ⏱️ Durée d’entraînement
- ~8 minutes (8 epochs)

## 📂 Fichiers générés

- `model4_model.h5` : modèle sauvegardé
- `accuracy_curve.png`, `loss_curve.png` : courbes
- `classification_report.txt` : rapport
- `confusion_matrix.png` : matrice

## Lancement

```bash
python train_model4.py --epochs 40 --batch_size 16 --unfreeze_layers 40