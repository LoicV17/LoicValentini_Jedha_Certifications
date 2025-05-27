FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ✅ C’est ici que tu copies tous les fichiers de ton app, y compris app.py
COPY . .

EXPOSE 7860

# ✅ Syntaxe correcte et fiable
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]