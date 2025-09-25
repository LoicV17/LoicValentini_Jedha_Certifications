# Utiliser une image Python légère
FROM python:3.9-slim

# Définir le dossier de travail
WORKDIR /code

# Copier les fichiers nécessaires
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

# Exposer le port attendu par HF (7860)
EXPOSE 7860

# Lancer Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
