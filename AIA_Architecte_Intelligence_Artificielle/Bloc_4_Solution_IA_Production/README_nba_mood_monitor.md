# 🏀 NBA Mood Monitor – Lakers Edition  
> **Un pipeline MLOps complet pour analyser en continu l’humeur des fans des Los Angeles Lakers sur Reddit**

---

## 🎯 Objectif du projet

Ce projet met en place un **pipeline Data & MLOps complet**, capable de collecter automatiquement les posts du subreddit **r/Lakers**, de les analyser grâce à des modèles NLP et des algorithmes de clustering, puis de les exposer dans un **dashboard interactif** pour suivre l’évolution de l’humeur et des thématiques de discussion de la communauté.

Ce pipeline assure :

- **Collecte automatisée** des posts Reddit (toutes les heures via Airflow)  
- **Nettoyage et normalisation** du texte  
- **Scoring émotionnel** via modèles Transformers  
- **Clustering non supervisé** des thématiques  
- **Suivi de la qualité des données et dérives**  
- **Stockage analytique** dans NeonDB  
- **Dashboard Streamlit** en 7 pages  
- **Orchestration Airflow**, **CI/CD Jenkins**, **infrastructure Docker**

L’objectif final : fournir une **vision continue, traçable et interprétable** de l’activité et de l’humeur des fans.

---

## ⚙️ Stack technique

| Domaine | Technologie | Rôle |
|--------|-------------|------|
| **Conteneurisation** | Docker / docker-compose | Environnements reproductibles |
| **Orchestration** | Apache Airflow | Scheduling + automatisation du pipeline |
| **Ingestion** | Reddit API | Récupération incrémentale des posts |
| **NLP / ML** | Transformers (HuggingFace) | Scoring émotionnel |
| **Clustering** | Sentence-Transformers + DBSCAN/KMeans | Détection des thématiques |
| **Base de données** | NeonDB (PostgreSQL cloud) | Stockage OLTP/analytique |
| **Dashboard** | Streamlit | Visualisation des émotions & clusters |
| **CI/CD** | Jenkins | Linting, tests, build docker |
| **Monitoring** | Logs Airflow + métriques internes | Qualité, dérives, stabilité |

---

## 🧩 Architecture du pipeline

Reddit API (r/lakers)
│
▼
[ Airflow DAGs ]
├── Fetch posts
├── Clean & normalize
├── Score sentiments (Transformers)
├── Embed + cluster (DBSCAN / KMeans)
├── Store in NeonDB
└── Compute drift & monitoring metrics
│
▼
NeonDB (PostgreSQL)
│
└── Streamlit Dashboard (7 pages)

---

## 🚀 Étapes principales du pipeline

| Étape | Description | Technologie |
|-------|-------------|-------------|
| 1️⃣ **Ingestion** | Récupération incrémentale des posts Reddit via API | Python, requests |
| 2️⃣ **Nettoyage** | Normalisation texte, suppression bruit, validation | pandas |
| 3️⃣ **Scoring émotionnel** | Modèle Transformers (positive / neutral / negative) | HuggingFace |
| 4️⃣ **Embedding** | Sentence-Transformers → vecteurs 768D | Python |
| 5️⃣ **Clustering** | DBSCAN ou KMeans (thèmes dominants) | sklearn |
| 6️⃣ **Stockage** | Insertion dans NeonDB | SQLAlchemy |
| 7️⃣ **Monitoring interne** | Stats, dérives, volumes | Python |
| 8️⃣ **Dashboard** | Visualisation multi-pages sur Streamlit | Python |
| 9️⃣ **CI/CD** | Lint, tests, build docker | Jenkins |

---

## 🖥️ Dashboard Streamlit – Contenu des 7 pages

1. **🏠 Home** – Vue globale du pipeline, derniers runs, KPIs clés  
2. **📅 Timeline** – Évolution temporelle des émotions  
3. **🧭 Clusters** – Regroupements thématiques + mots-clés  
4. **⚠️ Anomalies / Drift** – Comportements inhabituels, dérives, bruit  
5. **💬 Post Explorer** – Analyse détaillée d’un post (sentiment + cluster)  
6. **🛠 Logs & Pipeline** – Statuts Airflow + historique  
7. **📊 Data Quality** – Duplicats, taux d'erreur, distributions textuelles  

---

## 🔧 Installation locale

### 1️⃣ Cloner le projet
```bash
git clone https://github.com/loicvalentini/nba_mood_monitor_lakers.git
cd nba_mood_monitor_lakers
```
### 2️⃣ Créer et activer un environnement virtuel
 Créer et activer un environnement virtuel

```bash
python -m venv venv
```

```bash
source venv/bin/activate       # Linux / macOS
venv\Scripts\activate           # Windows
```

```bash
pip install -r requirements.txt
```

### 3️⃣ Configurer les variables d’environnement .env

Créer un fichier .env à la racine :

NEON_DB_URI=postgresql+psycopg2://user:password@host/dbname
REDDIT_CLIENT_ID=xxxx
REDDIT_SECRET=xxxx
REDDIT_USER_AGENT=nba-mood-monitor


(Optionnel si utilisation HuggingFace)

HF_TOKEN=your_huggingface_token

### 4️⃣ Lancer Airflow, Jenkins & le Dashboard

```bash
docker-compose up --build
```

### Accès aux services :

Airflow → http://localhost:8085
Jenkins → http://localhost:9090
Dashboard Streamlit → http://localhost:8501


### Loïc Valentini
Data Engineer & Machine Learning Architect
📍 Lille, France
🔗 LinkedIn : https://www.linkedin.com/in/loicvalentini
