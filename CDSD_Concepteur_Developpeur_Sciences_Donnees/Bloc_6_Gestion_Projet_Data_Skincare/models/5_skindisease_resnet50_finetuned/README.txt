README.txt – Modèle 5 : ResNet50 Fine-tuné

📁 Dossier : 5_skindisease_resnet50_finetuned/
📄 Script principal : train_model5.py
📦 Modèle utilisé : ResNet50 (couches supérieures dégelées)
🔍 Objectif : Classification binaire des lésions cutanées (bénigne vs maligne)
🕒 Durée d’entraînement : ~1h sur 10 époques (early stopping recommandé)
🔧 Paramètres :

Nombre d’époques : 40

Batch size : 16

Fine-tuning activé : Oui (certaines couches dégelées)

📊 Données :
Entraînement : 1975 images

Validation : 659 images

Test : 658 images

Classes : 2 (Benign, Malignant)

✅ Résultats :
Epoch avec meilleure validation (epoch 2) :
Accuracy (train) : 92.62%

AUC (train) : 97.77%

Loss (train) : 0.2012

Accuracy (val) : 88.32%

AUC (val) : 95.55%

Loss (val) : 0.2666

Modèle sauvegardé sous : model5_best_model.h5

Observations :
Le modèle converge rapidement, mais commence à sur-apprendre dès l’époque 3.

Les performances sur le set de validation restent stables autour de 87-88% d’accuracy et ~0.95 d’AUC jusqu’à epoch 5.

