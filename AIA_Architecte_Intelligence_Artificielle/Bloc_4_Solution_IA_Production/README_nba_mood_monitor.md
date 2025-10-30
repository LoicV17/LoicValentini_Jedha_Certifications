# 🏀 NBA Mood Monitor – Lakers Edition  
> Full MLOps Pipeline for Real-Time Fan Sentiment Analysis on Reddit  

## 🎯 Objectif du projet
Ce projet a pour objectif de construire un **pipeline MLOps complet** capable d’automatiser la collecte, le traitement, l’analyse et le suivi des émotions exprimées par les fans des **Los Angeles Lakers** sur Reddit.  

L’application met en œuvre un ensemble d’outils et de modèles de Machine Learning pour :
- Collecter en temps réel les messages du subreddit [`r/lakers`](https://www.reddit.com/r/lakers)
- Analyser leur **pertinence**, leur **thématique** (actualité vs discussion générale), et leur **sentiment**
- Centraliser les résultats dans une base de données **OLAP (NeonDB)**  
- Publier un **dashboard interactif (Tableau / Power BI)**  
- Surveiller la dérive du modèle avec **Evidently**  
- Automatiser les déploiements et mises à jour via **Airflow + Jenkins + Docker**

## ⚙️ Stack technique

| Domaine | Outil / Technologie | Rôle |
|----------|--------------------|------|
| **Data Ingestion** | Reddit API (JSON) | Récupération en temps réel des posts |
| **Orchestration** | Apache Airflow | Automatisation des tâches (fetch → analyse → monitoring) |
| **Machine Learning / NLP** | 🤖 Transformers (Hugging Face) | Sentiment & topic classification (DistilBERT + BART) |
| **Stockage** | NeonDB (PostgreSQL Cloud) | Base OLAP connectée à Tableau |
| **API de service** | FastAPI + Docker | Exposition du modèle en REST API |
| **CI/CD** | Jenkins | Build & test automatique à chaque commit |
| **Monitoring** | Evidently AI | Suivi de dérive du modèle et de la data |
| **Reporting** | Tableau / Power BI | Dashboard d’humeur et d’activité des fans |

## 🧩 Architecture du projet

```
Reddit API (r/lakers)
      │
      ▼
  [ Airflow DAGs ]
      ├── Fetch Reddit posts
      ├── Filter relevance (Lakers / not Lakers)
      ├── Classify topic (current / general)
      ├── Analyze sentiment (DistilBERT)
      ├── Save to NeonDB
      ├── Monitor drift (Evidently)
      └── Retrain if needed
      │
      ▼
NeonDB (PostgreSQL Cloud)
      │
      ├── Tableau Dashboard → Fan Mood Visualization
      └── Evidently Dashboard → Model Drift Reports
```

## 📂 Structure du workspace

```
nba_mood_monitor_lakers/
│
├── dags/
├── src/
│   ├── ingestion/
│   ├── models/
│   ├── preprocessing/
│   ├── database/
│   └── monitoring/
├── api/
├── ci_cd/
├── data/
├── dashboards/
├── models/
├── notebooks/
├── .env
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## 🚀 Étapes principales du pipeline

| Étape | Description | Technologie |
|--------|--------------|-------------|
| 1️⃣ **Collecte des données** | Extraction des messages via Reddit API | Python + requests |
| 2️⃣ **Nettoyage et prétraitement** | Suppression doublons, normalisation texte | pandas |
| 3️⃣ **Filtrage de pertinence** | Zero-shot : Lakers-related / Not Lakers | BART MNLI |
| 4️⃣ **Classification thématique** | Current event / General topic | BART MNLI |
| 5️⃣ **Analyse de sentiment** | POSITIVE / NEGATIVE / NEUTRAL | DistilBERT |
| 6️⃣ **Stockage OLAP** | Sauvegarde dans NeonDB (PostgreSQL) | SQLAlchemy |
| 7️⃣ **Dashboard utilisateur** | Visualisation “Fan Mood” | Tableau / Power BI |
| 8️⃣ **Monitoring** | Détection de dérive (drift lexical / sentiment) | Evidently |
| 9️⃣ **Déploiement API** | FastAPI + Docker + Jenkins | REST / CI/CD |
| 🔁 **Retrain automatique** | Si dérive détectée | Airflow DAG |

## 🔧 Installation locale

### 1️⃣ Cloner le projet
```bash
git clone https://github.com/loicvalentini/nba_mood_monitor_lakers.git
cd nba_mood_monitor_lakers
```

### 2️⃣ Créer un environnement virtuel
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3️⃣ Configurer les variables d’environnement `.env`
```
NEON_DB_URI=postgresql+psycopg2://user:password@host/main
HF_TOKEN=your_huggingface_token
```

### 4️⃣ Lancer les services Docker (API + Airflow + Jenkins)
```bash
docker-compose up --build
```

## 🧩 Monitoring Data Scientist (Evidently)
- Comparaison jour J vs J-7 des sentiments et du vocabulaire
- Dérive lexicale → `TextDriftMetric`
- Génération d’un rapport HTML : `/data/reports/drift_YYYYMMDD.html`
- Stockage des scores de drift dans la table `drift_metrics`

## 👨‍💻 Auteur
**Loïc Valentini**  
*Data Engineer & Machine Learning Architect*  
📍 Lille, France  
💼 [LinkedIn](https://www.linkedin.com/in/loicvalentini)

## 🧾 Licence
Projet sous licence MIT – libre d’utilisation et de modification à des fins non commerciales.
