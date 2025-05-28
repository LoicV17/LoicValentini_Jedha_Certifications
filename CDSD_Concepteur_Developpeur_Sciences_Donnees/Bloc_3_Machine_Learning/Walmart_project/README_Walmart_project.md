# 🛒 Analyse des ventes Walmart – Projet Data Science (Bloc 3 – CDSD)

## 📌 Objectif

Ce projet vise à analyser les ventes hebdomadaires des magasins Walmart à travers les États-Unis, afin d’identifier les facteurs influençant la performance commerciale et de prédire les ventes futures à l’aide de modèles de Machine Learning. Ce projet s’inscrit dans le cadre du **Bloc 3 – Machine Learning sur données structurées** de la certification **CDSD – Concepteur Développeur en Sciences des Données** (Jedha / RNCP 35288).

## 📁 Structure du projet

- `Walmart_Store_sales.csv` : Jeu de données initial contenant les ventes par magasin et département.
- `Project_instructions.ipynb` : Cahier des charges du projet.
- `Walmart_project_1.ipynb` : Analyse exploratoire et préparation des données.
- `Walmart_simplified_dataset.csv` : Jeu de données nettoyé et simplifié pour le Machine Learning.
- `Walmart_project_2.ipynb` : Modélisation, évaluation et sélection de modèles prédictifs.
- `README.md` : Ce document.

## 🧰 Technologies utilisées

- Python 3.x
- Pandas, NumPy
- Seaborn, Matplotlib
- Scikit-learn (modélisation)
- Jupyter Notebook

## 🧪 Méthodologie

### 📊 Analyse exploratoire (Notebook 1)

- Analyse temporelle et par type de magasin (A, B, C)
- Étude de l’impact de variables comme les vacances, les événements, ou les conditions économiques (CPI, taux de chômage)
- Visualisation des tendances de vente globales et spécifiques
- Nettoyage du dataset et simplification des colonnes pertinentes

### 🤖 Modélisation (Notebook 2)

- Objectif : Prédire les ventes hebdomadaires (`Weekly_Sales`)
- Techniques appliquées :
  - Régression linéaire
  - Régression ridge avec recherche d’hyperparamètres (`GridSearchCV`)
  - Régression lasso avec recherche d’hyperparamètres (`GridSearchCV`)
  - Comparaison des scores `R²`

## 📈 Résultats clés

- Les meilleurs scores de R² en entraînement sont obtenus avec les modèles **Ridge** et **Lasso** optimisés, dépassant **0.94**. Ces modèles montrent toutefois un léger **sur-apprentissage** (R² ≈ 0.975 en entraînement).
- Cet **overfitting modéré** ne semble pas altérer significativement les performances pratiques des modèles.
- La **variable “store”** ressort systématiquement comme la plus influente dans l’explication des ventes. Les variables globales comme `holiday_flag`, `temperature`, ou `fuel_price` ont un impact marginal.
- Le **modèle de régression linéaire simple**, bien qu’élémentaire, présente des performances comparables aux versions optimisées. Il pourrait suffire pour un usage en production selon les contraintes de complexité ou de maintenance.


## 👨‍🎓 Réalisé par

**Loic Valentini**  
Projet mené dans le cadre du Bloc 3 – Machine Learning sur données structurées  
Certification CDSD (Concepteur Développeur en Sciences des Données – Jedha / RNCP 35288)

## 📬 Contact

Pour toute question ou suggestion, vous pouvez me contacter via mon [profil LinkedIn](https://www.linkedin.com/in/loic-valentini-0a6238107/)
