# ========================================
# 📦 1. Imports
# ========================================
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, RandomizedSearchCV, ParameterSampler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    precision_recall_curve
)
import xgboost as xgb
import joblib
from pathlib import Path
import numpy as np
from tqdm import tqdm


from dotenv import load_dotenv
from pathlib import Path
import os

# ========================================
# 🔐 1bis. Charger secrets (connexion AWS)
# ========================================
env_path = Path(__file__).parent / "secrets" / ".env"
load_dotenv(dotenv_path=env_path)

s3_uri = os.getenv("MLFLOW_S3_BUCKET")
print(f"✅ Bucket MLflow configuré : {s3_uri}")

# ========================================
# 📥 2. Charger le dataset
# ========================================
file_path = "src/machine_learning_src.csv"
df = pd.read_csv(file_path)

print("✅ Données chargées avec succès")
print("Aperçu du dataset :")
print(df.head())

# ========================================
# 🧼 3. Préparer les données
# ========================================
df["gender"] = df["gender"].map({"F": 0, "M": 1})

label_encoder_category = LabelEncoder()
df["category"] = label_encoder_category.fit_transform(df["category"])

label_encoder_state = LabelEncoder()
df["state"] = label_encoder_state.fit_transform(df["state"])

print("\nTypes de colonnes après encodage :")
print(df.dtypes)

# ========================================
# 🎯 4. Définir X (features) et y (cible)
# ========================================
X = df.drop(columns=["is_fraud"])
y = df["is_fraud"]

print("\nDimensions de X et y :")
print("X :", X.shape)
print("y :", y.shape)

# ========================================
# ✂️ 5. Séparer en train et test
# ========================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print("\nTaille des jeux de données :")
print("X_train :", X_train.shape)
print("X_test  :", X_test.shape)

# ========================================
# ⚙️ 6. Définir l’espace de recherche RandomizedSearch
# ========================================
param_dist = {
    "n_estimators": np.arange(100, 500, 50),
    "max_depth": np.arange(3, 12, 2),
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "min_child_weight": [1, 5, 10],
    "scale_pos_weight": [1, 5, 10]
}

xgb_model = xgb.XGBClassifier(
    eval_metric="logloss",
    random_state=42
)

random_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_dist,
    n_iter=10,  # nombre d’essais aléatoires
    scoring="f1",
    cv=3,
    verbose=2,
    n_jobs=-1,
    random_state=42
)

# ========================================
# 📊 7. Suivi MLflow
# ========================================
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
mlflow.set_experiment("FraudDetection_XGBoost")

with mlflow.start_run(run_name="XGBoost_Fraud_RandomSearch"):

    # 🔹 Entraînement
    random_search.fit(X_train, y_train)

    best_params = random_search.best_params_
    print("\n✅ Meilleurs hyperparamètres :", best_params)

    # 🔹 Prédictions
    y_pred = random_search.predict(X_test)
    y_pred_proba = random_search.predict_proba(X_test)[:, 1]

    # ========================================
    # 📈 Rapport de classification
    # ========================================
    report_text = classification_report(y_test, y_pred)
    print("\n📈 Rapport de classification (test) :")
    print(report_text)

    # Sauvegarder rapport dans un fichier texte
    Path("models").mkdir(exist_ok=True)
    report_path = "models/classification_report.txt"
    with open(report_path, "w") as f:
        f.write(report_text)
    print(f"✅ Rapport de classification sauvegardé dans {report_path}")

    # ========================================
    # 📉 Matrice de confusion
    # ========================================
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Non fraude", "Fraude"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Matrice de confusion - XGBoost")
    cm_path = "models/confusion_matrix.png"
    plt.savefig(cm_path, bbox_inches="tight")
    print(f"✅ Matrice de confusion sauvegardée dans {cm_path}")

    # ========================================
    # 📉 Courbe ROC-AUC
    # ========================================
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("Courbe ROC - XGBoost")
    plt.legend(loc="lower right")
    roc_path = "models/roc_curve.png"
    plt.savefig(roc_path, bbox_inches="tight")
    print(f"✅ Courbe ROC sauvegardée dans {roc_path}")

    # ========================================
    # 📉 Courbe Precision-Recall
    # ========================================
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)

    plt.figure()
    plt.plot(recall, precision, color="blue", lw=2, label="PR curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Courbe Precision-Recall - XGBoost")
    plt.legend(loc="upper right")
    pr_path = "models/pr_curve.png"
    plt.savefig(pr_path, bbox_inches="tight")
    print(f"✅ Courbe Precision-Recall sauvegardée dans {pr_path}")

    # ========================================
    # 📌 Log MLflow
    # ========================================
    mlflow.log_params(best_params)

    # Rapport classification sous dict
    report_dict = classification_report(y_test, y_pred, output_dict=True)

    # Log métriques pour la classe fraude (1)
    mlflow.log_metric("precision_fraud", report_dict["1"]["precision"])
    mlflow.log_metric("recall_fraud", report_dict["1"]["recall"])
    mlflow.log_metric("f1_fraud", report_dict["1"]["f1-score"])

    # Log métriques pour la classe non-fraude (0)
    mlflow.log_metric("precision_nonfraud", report_dict["0"]["precision"])
    mlflow.log_metric("recall_nonfraud", report_dict["0"]["recall"])
    mlflow.log_metric("f1_nonfraud", report_dict["0"]["f1-score"])

    # Log moyennes
    mlflow.log_metric("f1_macro", report_dict["macro avg"]["f1-score"])
    mlflow.log_metric("f1_weighted", report_dict["weighted avg"]["f1-score"])

    # Log AUC
    mlflow.log_metric("roc_auc", roc_auc)

    # Log artifacts (rapports + plots)
    mlflow.log_artifact(report_path, artifact_path="reports")
    mlflow.log_artifact(cm_path, artifact_path="plots")
    mlflow.log_artifact(roc_path, artifact_path="plots")
    mlflow.log_artifact(pr_path, artifact_path="plots")

    # Sauvegarder modèle dans MLflow
    mlflow.sklearn.log_model(random_search.best_estimator_, artifact_path="model")

# ========================================
# 💾 8. Sauvegarde locale du modèle
# ========================================
local_model_path = "models/xgboost_fraud_model.pkl"
joblib.dump(random_search.best_estimator_, local_model_path)
print(f"✅ Modèle sauvegardé localement sous {local_model_path}")
