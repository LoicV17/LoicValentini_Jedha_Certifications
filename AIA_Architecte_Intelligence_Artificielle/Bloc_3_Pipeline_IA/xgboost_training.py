# ========================================
# 📦 1. Imports
# ========================================
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
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
import boto3
from botocore.exceptions import NoCredentialsError, ClientError

from dotenv import load_dotenv
import os

# ========================================
# 🔐 2. Charger secrets (connexion AWS)
# ========================================
env_path = Path(__file__).parent / "secrets" / ".env"
load_dotenv(dotenv_path=env_path)

s3_uri = os.getenv("MLFLOW_S3_BUCKET")
print(f"✅ Bucket MLflow configuré : {s3_uri}")

# ========================================
# 🔍 3. Vérifier connexion AWS / S3
# ========================================
try:
    aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    aws_region = os.getenv("AWS_DEFAULT_REGION", "eu-west-3")

    if not s3_uri:
        raise ValueError("❌ Variable d'environnement MLFLOW_S3_BUCKET manquante")

    if s3_uri.startswith("s3://"):
        bucket_name = s3_uri.replace("s3://", "")
    else:
        bucket_name = s3_uri

    s3_client = boto3.client(
        "s3",
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_key,
        region_name=aws_region
    )

    s3_client.head_bucket(Bucket=bucket_name)
    print(f"✅ Connexion réussie à S3 et accès au bucket '{bucket_name}'")

except NoCredentialsError:
    print("❌ Impossible de trouver les credentials AWS")
except ClientError as e:
    print(f"❌ Erreur de connexion au bucket : {e}")
except Exception as e:
    print(f"❌ Problème AWS : {e}")

# ========================================
# 📥 4. Charger le dataset
# ========================================
base_dir = Path(__file__).parent
file_path = base_dir / "src" / "machine_learning_src.csv"

df = pd.read_csv(file_path)

df = pd.read_csv(file_path)

# Supprimer la colonne inutile
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])


print("✅ Données chargées avec succès")
print("Aperçu du dataset :")
print(df.head())

# ========================================
# 🧼 5. Préparer les données
# ========================================
# Mapping simple pour gender
df["gender"] = df["gender"].map({"F": 0, "M": 1})

print("\nTypes de colonnes avant encodage OneHot:")
print(df.dtypes)

# ========================================
# 🎯 6. Définir X (features) et y (cible)
# ========================================
X = df.drop(columns=["is_fraud"])
y = df["is_fraud"]

print("\nDimensions de X et y :")
print("X :", X.shape)
print("y :", y.shape)

# ========================================
# ✂️ 7. Séparer en train et test
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
# ⚙️ 8. Pipeline avec OneHotEncoder
# ========================================
categorical_cols = ["category", "state"]
numeric_cols = [col for col in X.columns if col not in categorical_cols]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numeric_cols)
    ]
)

xgb_model = xgb.XGBClassifier(eval_metric="logloss", random_state=42)

pipeline = Pipeline(steps=[("preprocessor", preprocessor),
                           ("model", xgb_model)])

param_dist = {
    "model__n_estimators": np.arange(100, 500, 50),
    "model__max_depth": np.arange(3, 12, 2),
    "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
    "model__subsample": [0.6, 0.8, 1.0],
    "model__colsample_bytree": [0.6, 0.8, 1.0],
    "model__min_child_weight": [1, 5, 10],
    "model__scale_pos_weight": [1, 5, 10]
}

random_search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_dist,
    n_iter=100,
    scoring="f1",
    cv=3,
    verbose=2,
    n_jobs=-1,
    random_state=42
)

# ========================================
# 📊 9. Suivi MLflow
# ========================================
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
mlflow.set_experiment("FraudDetection_XGBoost")

with mlflow.start_run(run_name="XGBoost_Fraud_RandomSearch_OneHot"):

    random_search.fit(X_train, y_train)

    best_params = random_search.best_params_
    print("\n✅ Meilleurs hyperparamètres :", best_params)

    y_pred = random_search.predict(X_test)
    y_pred_proba = random_search.predict_proba(X_test)[:, 1]

    # 📈 Rapport classification
    report_text = classification_report(y_test, y_pred)
    print("\n📈 Rapport de classification (test) :")
    print(report_text)

    Path("models").mkdir(exist_ok=True)
    report_path = "models/classification_report.txt"
    with open(report_path, "w") as f:
        f.write(report_text)

    # 📉 Matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Non fraude", "Fraude"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Matrice de confusion - XGBoost")
    cm_path = "models/confusion_matrix.png"
    plt.savefig(cm_path, bbox_inches="tight")

    # 📉 Courbe ROC-AUC
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

    # 📉 Courbe Precision-Recall
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    plt.figure()
    plt.plot(recall, precision, color="blue", lw=2, label="PR curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Courbe Precision-Recall - XGBoost")
    plt.legend(loc="upper right")
    pr_path = "models/pr_curve.png"
    plt.savefig(pr_path, bbox_inches="tight")

    # 📌 Log MLflow
    mlflow.log_params(best_params)
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    mlflow.log_metric("precision_fraud", report_dict["1"]["precision"])
    mlflow.log_metric("recall_fraud", report_dict["1"]["recall"])
    mlflow.log_metric("f1_fraud", report_dict["1"]["f1-score"])
    mlflow.log_metric("precision_nonfraud", report_dict["0"]["precision"])
    mlflow.log_metric("recall_nonfraud", report_dict["0"]["recall"])
    mlflow.log_metric("f1_nonfraud", report_dict["0"]["f1-score"])
    mlflow.log_metric("f1_macro", report_dict["macro avg"]["f1-score"])
    mlflow.log_metric("f1_weighted", report_dict["weighted avg"]["f1-score"])
    mlflow.log_metric("roc_auc", roc_auc)

    mlflow.log_artifact(report_path, artifact_path="reports")
    mlflow.log_artifact(cm_path, artifact_path="plots")
    mlflow.log_artifact(roc_path, artifact_path="plots")
    mlflow.log_artifact(pr_path, artifact_path="plots")

    mlflow.sklearn.log_model(random_search.best_estimator_, artifact_path="model")

# ========================================
# 💾 10. Sauvegarde locale du modèle
# ========================================
local_model_path = "models/xgboost_fraud_model_onehot.pkl"
joblib.dump(random_search.best_estimator_, local_model_path)
print(f"✅ Modèle sauvegardé localement sous {local_model_path}")
