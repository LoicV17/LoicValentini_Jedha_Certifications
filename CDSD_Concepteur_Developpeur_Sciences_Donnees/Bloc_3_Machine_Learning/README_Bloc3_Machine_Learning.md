# 🤖 Bloc 3 – Machine Learning sur données structurées

Ce dossier contient trois projets réalisés dans le cadre du **Bloc 3 – Machine Learning sur données structurées** de la certification **CDSD – Concepteur Développeur en Sciences des Données** (Jedha / RNCP 35288).  
Ce bloc vise à maîtriser les techniques d'apprentissage supervisé et non supervisé, depuis la préparation des données jusqu’à l’interprétation des résultats.

---

## 🧪 Projets inclus

### 🔹 1. The North Face – Amélioration des ventes e-commerce

Projet de machine learning **non supervisé** visant à améliorer l’e-commerce de The North Face via une meilleure structuration du catalogue et des recommandations de produits.

#### 🎯 Objectif général
Optimiser la présentation des produits et générer des recommandations similaires à partir des descriptions textuelles uniquement.

#### 📌 Volets techniques

- **Clustering** de descriptions produits (TF-IDF + DBSCAN)
- **Système de recommandation** par similarité de cluster
- **Topic Modeling** via LSA (TruncatedSVD) pour extraire les thèmes marketing dominants

#### ✅ Résultats

| Volet            | Algorithmes                | Objectif atteint                                           |
|------------------|----------------------------|-------------------------------------------------------------|
| Clustering       | TF-IDF + LSA + DBSCAN      | +40 groupes sémantiques (matière, usage, coupe…)           |
| Recommandation   | Règle simple par cluster   | Produits similaires proposés sans données utilisateur       |
| Topic Modeling   | TF-IDF + SVD               | 5 grands thèmes produits identifiés                        |

#### 💼 Recommandations Business

- Menus de navigation par univers (cluster)
- Suggestions contextuelles sur les fiches produit
- Structuration éditoriale et campagnes autour des thèmes clés : matière durable, entretien, coupe, etc.

📁 Dossier : `North_face_project/`

---

### 🔹 2. Conversion Rate Challenge – Prédiction d’abonnement

Projet de classification supervisée visant à prédire si un utilisateur s’abonnera à une newsletter, à partir de données comportementales.

#### 📌 Méthodologie

- Analyse exploratoire et traitement des déséquilibres
- Comparaison de plusieurs modèles : **Logistic Regression**, **Random Forest**, **XGBoost**
- Optimisation via **GridSearch** & **cross-validation**
- Évaluation par **F1-score**, **precision**, **recall**

#### ✅ Résultats

- Modèle optimal : **Logistic Regression multivariée**
- F1-score sur le test : **0.77**
- Campagnes recommandées : ciblage par heure, type d’appareil, clics

📁 Dossier : `Conversion_rate_project/`

---

### 🔹 3. Walmart – Prévision des ventes hebdomadaires

Projet de régression supervisée visant à prédire les ventes hebdomadaires pour différents magasins Walmart.

#### 📌 Méthodologie

- Agrégation et enrichissement des données : jours fériés, température, fuel
- Régression linéaire, Ridge, Lasso
- Sélection des meilleurs hyperparamètres par GridSearch
- Analyse de la variance expliquée et des limitations

#### ✅ Résultats

- Modèles Ridge & Lasso atteignent **R² > 0.94** (train), **~0.91** (test)
- Faible impact des variables économiques globales
- Importance forte de la variable `store`

📁 Dossier : `Walmart_project/`

---

## 🧰 Technologies & outils

- Python (Pandas, Scikit-learn, XGBoost)
- NLP : SpaCy, TF-IDF, LSA
- Visualisation : WordCloud, Matplotlib, Plotly
- GridSearchCV pour l’optimisation
- Git / GitHub pour le versioning

---

## 📚 Compétences mobilisées

- Prétraitement et vectorisation de données structurées et textuelles
- Apprentissage supervisé : classification & régression
- Apprentissage non supervisé : clustering & topic modeling
- Évaluation des modèles (F1-score, R², etc.)
- Visualisation des résultats & interprétation business
- Recommandation produit simple & segmentation thématique

---

👨‍🎓 Réalisé par **Loic Valentini**  
🔗 [LinkedIn](https://www.linkedin.com/in/loic-valentini-0a6238107/)  
🏫 Certification CDSD – Jedha Bootcamp  
📅 Année : 2025
