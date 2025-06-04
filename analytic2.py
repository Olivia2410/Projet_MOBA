import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple
import os
from collections import defaultdict

BASE_DIR = os.path.dirname(__file__)  # Dossier du script main.py
DATA_DIR = os.path.join(BASE_DIR, "static")

# === 1. Charger les prédictions ===
champion_winrates_df = pd.read_csv(os.path.join(DATA_DIR, "predicted_champion_winrates.csv"))
pair_winrates_df = pd.read_csv(os.path.join(DATA_DIR, "pair_winrate_predictions_with_embeddings.csv"))
champion_roles_df = pd.read_csv(os.path.join(DATA_DIR, "champion_roles.csv"))



# === Pondérations apprises depuis le modèle PyTorch ===
learned_weights = torch.tensor([0.5, 0.3, 0.2])  # winrate, synergy, counter
learned_bias = torch.tensor(0.05)


# === 2. WINRATE D'UN CHAMPION ===
def get_champion_winrate(champion: str) -> Optional[float]:
    """Retourne le winrate prédit d'un champion"""
    row = champion_winrates_df[champion_winrates_df["Champion"] == champion]
    return float(row["Predicted_Winrate"].values[0]) if not row.empty else None
"""
# === 3. SYNERGIE ENTRE 2 CHAMPIONS ===
def get_pair_synergy(champ1: str, champ2: str) -> Optional[float]:
    Calcule la synergie prédite entre deux champions
    champ1, champ2 = sorted([champ1, champ2])
    row = pair_winrates_df[
        (pair_winrates_df["Champion_1"] == champ1) & 
        (pair_winrates_df["Champion_2"] == champ2)
    ]
    return float(row["Predicted_Winrate"].values[0]) if not row.empty else None
"""

def get_pair_synergy(champ1: str, champ2: str) -> Optional[float]:
    """Calcule la synergie prédite entre deux champions, même s'ils n'ont jamais été vus ensemble"""
    champ1, champ2 = sorted([champ1, champ2])
    row = pair_winrates_df[
        (pair_winrates_df["Champion_1"] == champ1) & 
        (pair_winrates_df["Champion_2"] == champ2)
    ]
    
    if not row.empty:
        return float(row["Predicted_Winrate"].values[0])
    
    # Si pas de données, estimer par défaut
    winrate1 = get_champion_winrate(champ1)
    winrate2 = get_champion_winrate(champ2)
    
    if winrate1 is not None and winrate2 is not None:
        # Exemple d'heuristique : moyenne des deux winrates moins un petit biais d'incertitude
        estimated_synergy = (winrate1 + winrate2) / 2 - 0.01
        return round(estimated_synergy, 4)
    
    # Si les winrates sont indisponibles, retourner une valeur par défaut neutre
    return 0.5  # ou toute autre valeur que tu juges comme "neutre"

# ===  calculer les taux de sélection ===
def get_champion_pick_rates():
    """Calcule les taux de sélection des champions à partir des données de match"""
    df = pd.read_csv(os.path.join(DATA_DIR, "top_match_data.csv"))
    
    # Compter les apparitions de chaque champion
    pick_counts = defaultdict(int)
    roles = ['top', 'jungle', 'mid', 'adc', 'support']
    
    for _, row in df.iterrows():
        for role in roles:
            pick_counts[row[f'blue_{role}']] += 1
            pick_counts[row[f'red_{role}']] += 1
    
    total_picks = sum(pick_counts.values())
    
    # Calculer les taux en pourcentage
    pick_rates = {
        champ: (count / total_picks * 100) 
        for champ, count in pick_counts.items()
    }
    
    return pick_rates

def get_counter_score(champ1: str, champ2: str, depth: int = 1) -> Optional[dict]:
    """
    Calcule un score de matchup entre deux champions avec interprétation :
    - winrate relatif
    - score d’avantage
    - message d’analyse (style MobaChampion)
    """
    winrate_1 = get_champion_winrate(champ1)
    winrate_2 = get_champion_winrate(champ2)
    if winrate_1 is None or winrate_2 is None:
        return None

    direct_score = winrate_1 - winrate_2

    if depth > 1:
        indirect_scores = []
        all_champs = champion_winrates_df["Champion"].tolist()

        for intermediate in all_champs:
            if intermediate in [champ1, champ2]:
                continue
            score1 = get_counter_score(champ1, intermediate, depth=1)
            score2 = get_counter_score(intermediate, champ2, depth=1)

            if score1 and score2:
                indirect_scores.append((score1["advantage_score"] + score2["advantage_score"]) / 2)

        if indirect_scores:
            indirect_avg = np.mean(indirect_scores)
            direct_score = (direct_score + indirect_avg) / 2

    # Interprétation du score
    percent_advantage = round(50 + direct_score * 50, 2)  # transformation en pourcentage
    if percent_advantage > 55:
        analysis = f"{champ1} contre très bien {champ2} ({percent_advantage}% de winrate estimé)."
    elif percent_advantage > 52:
        analysis = f"{champ1} a un léger avantage sur {champ2}."
    elif percent_advantage < 45:
        analysis = f"{champ1} est généralement contré par {champ2}."
    elif percent_advantage < 48:
        analysis = f"{champ1} semble en difficulté contre {champ2}."
    else:
        analysis = f"{champ1} et {champ2} sont plutôt équilibrés dans ce matchup."

    return {
        "advantage_score": round(direct_score, 4),
        "percent_advantage": percent_advantage,
        #"analysis": analysis
    }

# === 5. SCORE DE COMPOSITION ===
def get_team_score(team_composition: List[str]) -> float:
    """Évalue la force globale d'une composition d'équipe"""
    if not team_composition:
        return 0.0
    
    # Winrate moyen
    winrates = [get_champion_winrate(champ) or 0 for champ in team_composition]
    avg_winrate = np.mean(winrates)
    
    # Synergie moyenne
    synergy_scores = []
    for i in range(len(team_composition)):
        for j in range(i+1, len(team_composition)):
            synergy = get_pair_synergy(team_composition[i], team_composition[j]) or 0
            synergy_scores.append(synergy)
    
    avg_synergy = np.mean(synergy_scores) if synergy_scores else 0
    
    # Score global pondéré
    return 0.7 * avg_winrate + 0.3 * avg_synergy

# === 6. RECOMMANDATION AMÉLIORÉE ===
def recommend_champions(
    team_compo: List[str],
    enemy_compo: List[str],
    top_n: int = 3,
    role: Optional[str] = None,
    counter_depth: int = 1
) -> List[Tuple[str, float]]:
    """
    Recommande des champions en fonction de la composition alliée et ennemie
    Args:
        team_compo: Liste des champions alliés
        enemy_compo: Liste des champions ennemis
        top_n: Nombre de recommandations à retourner
        role: Rôle du champion à recommander (optionnel)
        counter_depth: Profondeur d'analyse des counters (1 pour direct seulement)
    """
    scores = []
    all_champs = champion_winrates_df["Champion"].tolist()
    
    for champ in all_champs:
        if champ in team_compo or champ in enemy_compo:
            continue
            
        # Filtre par rôle si spécifié
        if role and not is_champion_role(champ, role):  # À implémenter
            continue
            
        base_wr = get_champion_winrate(champ) or 0
        
        # Synergie avec l'équipe
        synergy = np.mean([
            get_pair_synergy(champ, ally) or 0 
            for ally in team_compo
        ]) if team_compo else 0
        
        # Counter contre l'équipe ennemie
        counter = np.mean([
            get_counter_score(champ, enemy, counter_depth) or 0 
            for enemy in enemy_compo
        ]) if enemy_compo else 0
        
        # Score global
        features = torch.tensor([base_wr, synergy, counter], dtype=torch.float32)
        score = torch.dot(learned_weights, features) + learned_bias
        scores.append((champ, round(score.item(), 4)))
    
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_n]

# === 7. ANALYSE DE COMPOSITION ===
def analyze_compositions(
    team_compo: List[str],
    enemy_compo: List[str]
) -> Dict[str, float]:
    """Analyse approfondie des deux compositions"""
    team_score = get_team_score(team_compo)
    enemy_score = get_team_score(enemy_compo)
    
    # Score de contre-composition
    counter_scores = []
    for ally in team_compo:
        for enemy in enemy_compo:
            score = get_counter_score(ally, enemy) or 0
            counter_scores.append(score)
    
    avg_counter = np.mean(counter_scores) if counter_scores else 0
    
    return {
        "team_score": round(team_score, 4),
        "enemy_score": round(enemy_score, 4),
        "counter_advantage": round(avg_counter, 4),
        "predicted_win_probability": round(
            sigmoid(team_score - enemy_score + 0.2 * avg_counter), 4
        )
    }

def sigmoid(x: float) -> float:
    """Fonction sigmoïde pour convertir un score en probabilité"""
    return 1 / (1 + np.exp(-x))


# Créer un dictionnaire {champion: [role1, role2]}
champion_roles_map = {
    row["champion"]: row["roles"].split(",")
    for _, row in champion_roles_df.iterrows()
}

def get_champion_roles(champion: str) -> List[str]:
    """Retourne les rôles principaux d'un champion (depuis champion_roles.csv)"""
    return champion_roles_map.get(champion, [])

def is_champion_role(champion: str, role: str) -> bool:
    """Vérifie si un champion correspond à un rôle donné"""
    return role.lower() in get_champion_roles(champion)

# === Nouvelle fonction pour générer toutes les combinaisons possibles ===
def generate_all_team_compositions(player_champions: Dict[str, List[str]]) -> List[List[str]]:
    """
    Génère toutes les combinaisons possibles d'équipe à partir des champions que chaque joueur peut jouer.
    
    Args:
        player_champions: Dictionnaire où les clés sont les rôles (top, mid, jungle, etc.)
                         et les valeurs sont les listes de champions que le joueur peut jouer pour ce rôle.
    
    Returns:
        Liste de toutes les compositions d'équipe possibles (chaque composition est une liste de champions)
    """
    from itertools import product
    
    # Convertir le dictionnaire en listes de champions par rôle
    roles = list(player_champions.keys())
    champions_per_role = list(player_champions.values())
    
    # Générer toutes les combinaisons possibles
    all_combinations = list(product(*champions_per_role))
    
    # Convertir les tuples en listes
    return [list(combo) for combo in all_combinations]

# === Nouvelle fonction pour évaluer toutes les compositions ===
def evaluate_all_compositions(player_champions: Dict[str, List[str]], enemy_compo: List[str] = None) -> List[Dict]:
    """
    Évalue toutes les compositions possibles selon différents critères.
    
    Args:
        player_champions: Dictionnaire des champions que chaque joueur peut jouer
        enemy_compo: Composition ennemie (optionnelle, pour évaluer les counters)
    
    Returns:
        Liste de dictionnaires contenant les évaluations pour chaque composition
    """
    # Générer toutes les compositions possibles
    all_compositions = generate_all_team_compositions(player_champions)
    
    results = []
    
    for composition in all_compositions:
        # Calculer les scores pour cette composition
        team_score = get_team_score(composition)
        
        # Si on a une composition ennemie, calculer l'avantage de contre
        counter_advantage = 0
        if enemy_compo:
            counter_scores = []
            for ally in composition:
                for enemy in enemy_compo:
                    score = get_counter_score(ally, enemy) or {"advantage_score": 0}
                    counter_scores.append(score["advantage_score"])
            counter_advantage = np.mean(counter_scores) if counter_scores else 0
        
        # Calculer la synergie moyenne
        synergy_scores = []
        for i in range(len(composition)):
            for j in range(i+1, len(composition)):
                synergy = get_pair_synergy(composition[i], composition[j]) or 0
                synergy_scores.append(synergy)
        avg_synergy = np.mean(synergy_scores) if synergy_scores else 0
        
        # Calculer le winrate moyen
        winrates = [get_champion_winrate(champ) or 0 for champ in composition]
        avg_winrate = np.mean(winrates)
        
        results.append({
            "composition": composition,
            "team_score": round(team_score, 4),
            "avg_winrate": round(avg_winrate, 4),
            "avg_synergy": round(avg_synergy, 4),
            "counter_advantage": round(counter_advantage, 4) if enemy_compo else None,
            "overall_score": round(0.5*avg_winrate + 0.3*avg_synergy + 0.2*counter_advantage, 4) if enemy_compo 
                            else round(0.6*avg_winrate + 0.4*avg_synergy, 4)
        })
    
    return results

# === Nouvelle fonction pour recommander la meilleure équipe ===
def recommend_best_team(player_champions: Dict[str, List[str]], enemy_compo: List[str] = None, 
                        strategy: str = "overall") -> Dict:
    """
    Recommande la meilleure équipe selon différents critères.
    
    Args:
        player_champions: Dictionnaire des champions que chaque joueur peut jouer
        enemy_compo: Composition ennemie (optionnelle)
        strategy: Critère de sélection ("winrate", "synergy", "counter", "overall")
    
    Returns:
        Dictionnaire avec la meilleure composition et les scores
    """
    evaluations = evaluate_all_compositions(player_champions, enemy_compo)
    
    if not evaluations:
        return None
    
    if strategy == "winrate":
        best = max(evaluations, key=lambda x: x["avg_winrate"])
        best["criteria"] = "Meilleur winrate moyen"
    elif strategy == "synergy":
        best = max(evaluations, key=lambda x: x["avg_synergy"])
        best["criteria"] = "Meilleure synergie"
    elif strategy == "counter" and enemy_compo:
        best = max(evaluations, key=lambda x: x["counter_advantage"])
        best["criteria"] = "Meilleur contre à l'équipe ennemie"
    else:
        best = max(evaluations, key=lambda x: x["overall_score"])
        best["criteria"] = "Meilleur score global"
    
    return best