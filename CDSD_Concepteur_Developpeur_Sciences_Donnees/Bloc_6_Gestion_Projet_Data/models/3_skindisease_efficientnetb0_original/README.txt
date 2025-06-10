# Skin Disease Classification – EfficientNetB0 (gelé)

Ce modèle utilise EfficientNetB0 (pré-entraîné sur ImageNet) pour classer les lésions cutanées en deux classes : benign et malignant.

## ⚙️ Paramètres principaux

- ✅ Modèle : EfficientNetB0 (`include_top=False`)
- ❄️ Fine-tuning : désactivé (backbone entièrement gelé)
- 🔍 Prétraitement : `preprocess_input` EfficientNetB0
- 🎯 Tâche : classification binaire (sigmoid)

## 📁 Données utilisées

- Entraînement : `train_skindisease/` (1975 images)
- Validation : `val_skindisease/` (659 images)
- Test final : `test_skindisease/` (658 images)
- Format : `.jpg`, 2 classes (`benign`, `malignant`)

## 🧪 Résultats

- **val_accuracy max** : ✅ 84.8%
- **train_accuracy** : ~86.7%
- **val_loss** : en diminution constante jusqu'à ~0.32
- Aucun surapprentissage détecté

## ⏱️ Temps d'entraînement

- ~6 minutes pour 10 époques
- Batch size : 32

## 📂 Fichiers générés

- `model3_model.h5` : modèle sauvegardé
- `accuracy_curve.png`, `loss_curve.png` : courbes
- `classification_report.txt` : rapport
- `confusion_matrix.png` : matrice

## 🔍 Conclusion

Ce modèle offre un excellent compromis précision/temps. Il constitue une baseline robuste sans nécessité de fine-tuning. À recommander pour déploiement rapide ou comparaison avec des architectures plus lourdes.
