README.txt – Modèle 6 : EfficientNetB0 (gelé) – HAM10000

📁 Dossier : models/6_ham10000_efficientnetb0_original/
📄 Script principal : train_model6.py
📦 Architecture utilisée : EfficientNetB0 préentraînée (poids ImageNet, sans fine-tuning)
🎯 Objectif : Classification multiclasse (7 classes) de lésions cutanées du dataset HAM10000

📊 Données :
Entraînement : 6007 images

Validation : 2008 images

Format images : .jpg – redimensionnées à 224x224

Normalisation : faite via preprocess_input() (EfficientNet)

⚙️ Configuration du script :
🔁 ImageDataGenerator avec preprocessing_function=preprocess_input

❄️ Base EfficientNetB0 gelée (aucune couche dégelée)

🧮 class_weight automatique pour gérer le fort déséquilibre de classes

📦 Optimiseur : Adam

🎯 Fonction de perte : categorical_crossentropy

🔁 Callbacks : EarlyStopping(patience=5) + ModelCheckpoint

🧪 Batch size : 32 (via argparse)

⏳ Époques : 20 (early stopping activé)

✅ Résultats obtenus :
Epoch avec la meilleure val_accuracy : epoch 16

Accuracy (train) : 72.7%

Loss (train) : 0.6851

Accuracy (val) : 71.1%

Loss (val) : 0.8213

✅ Modèle sauvegardé : model6_bestmodel.h5
📈 Courbes sauvegardées : model6_accuracy.png et model6_loss.png
📊 Évaluation finale :

Rapport de classification → model6_classification_report.txt

Matrice de confusion → model6_confusion_matrix.png

🛠️ Remarques :
Le modèle converge correctement malgré les classes déséquilibrées.

La performance de ~71% en validation est très correcte pour un modèle EfficientNet gelé sans fine-tuning.

Des pistes pour amélioration future :

Dégel progressif des couches supérieures

Data augmentation plus poussée

Optimisation du learning rate