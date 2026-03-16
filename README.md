# MOBA Draft Analyzer

Outil d’analyse et de recommandation de compositions d’équipe pour le jeu **League of Legends**.

Ce projet combine **analyse de données**, **machine learning** et **développement web** afin d'aider à :

- analyser les **matchups entre champions**
- mesurer la **synergie d’équipe**
- recommander les **meilleurs picks pendant une draft**
- générer la **meilleure composition d’équipe possible**

Le système utilise des données issues de l’API de Riot Games et des modèles de machine learning avec **PyTorch**.

Contexte Spécifique à LoL
Notre projet porte sur League of Legends (LoL), le MOBA le plus populaire d´evelopp´e
par Riot Games. LoL repose sur :
- des matchs en **5v5**,
- une répartition en 5 rôles : **Top, Jungle, Mid, ADC, Support**,
- plus de **160 champions** aux capacités uniques,

---

# Fonctionnalités

## Analyse des champions

- Affichage des **winrates prédits**
- Calcul des **pick rates**
- Recherche et tri des champions

---

## Analyse de synergie

Calcul de la **synergie entre deux champions**.

Si la paire n’existe pas dans les données, une estimation est faite à partir du winrate individuel des champions.

---

## Analyse de matchup (counter)

Le système calcule l’avantage entre deux champions et fournit un score interprété comme une **probabilité de victoire**.

---

## Recommandation de champions

Le système recommande automatiquement des champions selon :

- le **winrate individuel**
- la **synergie avec l’équipe**
- les **counters contre l’équipe ennemie**

Le score final est calculé par :

```python
score = w1 * winrate + w2 * synergy + w3 * counter + bias
```

---

# Analyse de composition d’équipe

Le système peut :

- analyser une composition alliée
- analyser une composition ennemie
- prédire la **probabilité de victoire**

```python
P(win) = sigmoid(team_score - enemy_score + counter_advantage)
```

---

# Team Builder

Interface interactive permettant de :

- sélectionner les champions jouables par chaque joueur
- générer toutes les compositions possibles
- analyser :

  - winrate moyen
  - synergie d’équipe
  - capacité de counter

Le système peut recommander **la meilleure équipe possible** selon différentes stratégies :

- meilleur winrate
- meilleure synergie
- meilleur counter
- meilleur score global

---

# Structure du projet

```
metanalysis/
│
├── main.py
├── analytic.py
├── counter_score.py
├── winrate.py
├── synergy.py
│
├── templates/
│   ├── champions.html
│   ├── synergy.html
│   ├── counter.html
│   ├── recommend.html
│   └── team_builder.html
│
├── static/
│   ├── champion_lanes_summary.csv
│   ├── champion_roles.csv
│   ├── predicted_champion_winrates.csv
│   ├── pair_winrate_predictions_with_embeddings.csv
│   ├── top_match_data.csv
│
└── README.md
```

---

# Technologies utilisées

## Backend

- Python
- FastAPI
- PyTorch
- Pandas
- NumPy

## Frontend

- HTML
- CSS
- JavaScript
- Jinja2

---

# Installation

Cloner le projet :

```bash
git clone https://github.com/Olivia2410/Projet_MOBA.git
cd Projet_MOBA
```

Créer un environnement virtuel :

```bash
python -m venv venv
```

Activer l’environnement :

Linux / Mac

```bash
source venv/bin/activate
```

Windows

```bash
venv\Scripts\activate
```

Installer les dépendances :

```bash
pip install -r requirements.txt
```

---

# Lancer l’application

```bash
python main.py
```

Le serveur démarre avec **Uvicorn**.

Application disponible sur :

```
http://127.0.0.1:8000
```

---

# Pages disponibles

| Route | Description |
|------|-------------|
| /champions | Liste des champions avec winrate et pickrate |
| /synergy | Analyse de synergie entre deux champions |
| /counter | Analyse de matchup |
| /recommend | Recommandation de picks |
| /team_builder | Construction et analyse d’équipe |

---

# Améliorations possibles

- modèle deep learning pour prédiction de winrate
- embeddings de champions plus avancés
- visualisation graphique des synergies
- déploiement web (Docker / cloud)

---

# Auteur

Projet réalisé dans le cadre d’un projet universitaire en **UE PROJET** par **Lucile HU et Olivia ZHENG* en double licence Infromatique et Mathématiques durant l'année universitaire 2024-2025, projet encadré par **Mohamed OUAGUENOUNI**  .
