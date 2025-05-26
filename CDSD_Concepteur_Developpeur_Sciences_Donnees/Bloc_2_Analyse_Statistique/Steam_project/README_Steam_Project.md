# 🎮 Analyse du catalogue Steam – Projet Data Science

## 📌 Objectif

Ce projet vise à analyser en profondeur le catalogue de jeux vidéo de la plateforme Steam afin d'identifier les facteurs influençant la popularité et les ventes des jeux. L'objectif est de fournir des insights stratégiques pour le développement et la publication de jeux sur Steam.

## 📁 Structure du projet

- `s3://full-stack-bigdata-datasets/Big_Data/Project_Steam/steam_game_output.json` : Lien du dataset
- `Steam_exploration.ipynb` : Copie du notebook, les visualisation ne sont visibles que sur Databricks ou dans le dossier "plots"
- `https://databricks-prod-cloudfront.cloud.databricks.com/public/4027ec902e239c93eaaa8714f173bcfc/3987453788770744/1535238523985856/4077541869256073/latest.html` : Lien public vers le notebook Databricks
- `README.md` : Ce document.

## 🧰 Technologies utilisées

- Python 3.x
- Apache Spark (PySpark)
- Databricks
- Pandas
- Matplotlib / Seaborn

## 📊 Analyses réalisées

1. Évolution des sorties de jeux dans le temps : Analyse des tendances de publication mensuelles et annuelles.
2. Répartition des genres : Identification des genres les plus représentés sur la plateforme.
3. Analyse des éditeurs : Étude des éditeurs les plus actifs et de leurs genres de prédilection.
4. Analyse des avis : Corrélation entre le nombre d'avis et la note moyenne des jeux.
5. Analyse des plateformes : Disponibilité des jeux sur Windows, Mac et Linux.
6. Analyse des revenus estimés : Estimation des revenus par éditeur en fonction du prix et du nombre d'avis.
7. Analyse des langues : Langues les plus représentées dans les jeux.

## 📈 Visualisations clés

- Graphiques en barres pour la répartition des genres et des langues.
- Heatmaps pour la relation éditeurs-genres.
- Graphiques en lignes pour l'évolution des sorties de jeux.
- Nuages de points pour l'analyse des avis.

## 📌 Conclusions principales

- La majorité des jeux sont disponibles sur Windows, avec une minorité sur Mac et Linux.
- Les genres les plus populaires sont Indie, Action et Adventure.
- Les éditeurs les plus actifs ne sont pas nécessairement les plus rentables.
- Une forte disparité de revenus existe entre les éditeurs.

## 📬 Contact

Pour toute question ou suggestion, n'hésitez pas à me contacter via mon profil GitHub.
