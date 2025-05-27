FROM python:3.9-slim

# Créer un utilisateur non-root
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user
WORKDIR $HOME/app

# Installer dépendances système
USER root
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Installer les dépendances Python
RUN pip install --upgrade pip && pip install mlflow pandas scikit-learn joblib

# Revenir à l’utilisateur normal
USER user

# Créer manuellement le dossier mlruns avec les bons droits
RUN mkdir -p $HOME/app/mlruns

# Copier le reste du projet (vide ou avec fichiers)
COPY --chown=user . .

EXPOSE 8501

# Lancer MLflow avec backend local
CMD ["mlflow", "ui", "--host", "0.0.0.0", "--port", "8501", "--backend-store-uri", "./mlruns"]
