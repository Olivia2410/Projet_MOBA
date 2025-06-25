import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple
import os
from collections import defaultdict
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from typing import List
import time

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
    """Retourne le winrate prédit d'un champion, insensible à la casse et aux espaces"""
    champion = champion.strip().lower()
    row = champion_winrates_df[champion_winrates_df["Champion"].str.lower().str.strip() == champion]
    return float(row["Predicted_Winrate"].values[0]) if not row.empty else None

# === 3. SYNERGIE ENTRE 2 CHAMPIONS ===
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

# === 4. calculer les taux de sélection ===
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

counter_data_df = pd.read_csv(os.path.join(DATA_DIR, "counter_score_predictions_with_embeddings.csv"))

def get_counter_score(champ1, champ2):
    champ1 = champ1.strip().lower()
    champ2 = champ2.strip().lower()

    df = counter_data_df.copy()
    df["Attacker"] = df["Attacker"].str.lower().str.strip()
    df["Defender"] = df["Defender"].str.lower().str.strip()

    match = df[(df["Attacker"] == champ1) & (df["Defender"] == champ2)]

    if not match.empty:
        advantage_score = match["Counter_Score"].values[0]
    else:
        winrate_1 = get_champion_winrate(champ1)
        winrate_2 = get_champion_winrate(champ2)
        if winrate_1 is None or winrate_2 is None:
            return None

        advantage_score = winrate_1 - winrate_2
    percent_advantage = round(50 + advantage_score * 50, 2)

    # Évaluation qualitative commune
    if percent_advantage >= 60:
        combat_score = "Excellente"
    elif percent_advantage >= 55:
        combat_score = "Bonne"
    elif percent_advantage >= 50:
        combat_score = "Correcte"
    elif percent_advantage >= 45:
        combat_score = "Faible"
    else:
        combat_score = "Très faible"

    return {
        "score": round(0.5 + advantage_score / 2, 4),
        "combat_score": combat_score,
        "percent_advantages": percent_advantage,
        "advantage_score": round(advantage_score, 4),
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

def recommend_champions(
    team_compo: List[str],
    enemy_compo: List[str],
    top_n: int = 3,
    role: Optional[str] = None,
) -> List[Tuple[str, float]]:
    scores = []
    all_champs = champion_winrates_df["Champion"].tolist()

    for champ in all_champs:
        if champ in team_compo or champ in (enemy_compo or []):
            continue

        if role and not is_champion_role(champ, role):
            continue

        base_wr = get_champion_winrate(champ)
        if base_wr is None:
            continue

        synergy = np.mean([
            get_pair_synergy(champ, ally) or 0
            for ally in team_compo
        ]) if team_compo else 0

        counter_scores = []
        if enemy_compo:
            for enemy in enemy_compo:
                result = get_counter_score(champ, enemy)
                if result:
                    counter_scores.append(result.get("advantage_score", 0))
        counter = np.mean(counter_scores) if counter_scores else 0

        features = torch.tensor([base_wr, synergy, counter], dtype=torch.float32)
        score = torch.dot(learned_weights, features) + learned_bias

        scores.append((champ, round(score.item(), 4)))

    scores.sort(key=lambda x: x[1], reverse=True)

    if not scores:
        print("⚠️ Aucune recommandation générée.")
        print(f"🔍 Rôle demandé : {role}")
        print(f"🧑‍🤝‍🧑 Équipe actuelle : {team_compo}")
        print(f"🎯 Ennemis : {enemy_compo}")
        print("🔎 Champions valides détectés pour le rôle :")
        for champ in all_champs:
            if champ not in team_compo and champ not in enemy_compo:
                if is_champion_role(champ, role):
                    print(f"✅ {champ} peut jouer {role}")

    return scores[:top_n]

# === 7. ANALYSE DE COMPOSITION ===
def analyze_compositions(
    team_compo: List[str],
    enemy_compo: List[str]
) -> Dict[str, float]:
    """Analyse approfondie des deux compositions"""
    
    # Calcul du score de l'équipe alliée
    team_score = get_team_score(team_compo) or 0.0
    
    # Calcul du score de l'équipe ennemie
    enemy_score = get_team_score(enemy_compo) or 0.0

    # Score de contre-composition corrigé
    counter_scores = []
    for ally in team_compo:
        for enemy in enemy_compo:
            result = get_counter_score(ally, enemy)
            if result:
                counter_scores.append(result.get("advantage_score", 0.0))

    avg_counter = np.mean(counter_scores) if counter_scores else 0.0

    # Prédiction de victoire avec ajustement selon l'avantage de counter
    predicted_win_probability = sigmoid(
        team_score - enemy_score + 0.2 * avg_counter
    )

    return {
        "team_score": round(team_score, 4),
        "enemy_score": round(enemy_score, 4),
        "counter_advantage": round(avg_counter, 4),
        "predicted_win_probability": round(predicted_win_probability, 4)
    }


def sigmoid(x: float) -> float:
    """Fonction sigmoïde pour convertir un score en probabilité"""
    return 1 / (1 + np.exp(-x))



# Crée un dictionnaire {champion: [role1, role2]} avec nettoyage .lower()
champion_roles_map = {
    row["champion"]: [r.strip().lower() for r in row["roles"].split(",")]
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
    Retourne la liste de toutes les compositions d'équipe possibles (chaque composition est une liste de champions)
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
    On a player_champions: Dictionnaire des champions que chaque joueur peut jouer
      et enemy_compo: Composition ennemie (optionnelle, pour évaluer les counters)
    
    Retourne une liste de dictionnaires contenant les évaluations pour chaque composition
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
    
    role_composition = {}
    for role, champions in player_champions.items():
        for champ in best["composition"]:
            if champ in champions:
                role_composition[role] = champ
                break
    return {
        "composition": role_composition,
        "avg_winrate": best["avg_winrate"],
        "avg_synergy": best["avg_synergy"],
        "counter_advantage": best.get("counter_advantage", 0),
        "overall_score": best["overall_score"],
        "criteria": best["criteria"]}

def get_player_mastery_champions(player_id: str, region: str, top_n: int = 3) -> List[str]:
    """
    Récupère les n champions les plus maîtrisés d'un joueur via xdx.gg
    
    Args:
        player_id (str): ID du joueur (ex: "cat")
        region (str): Code région (ex: "EUW", "NA1")
        top_n (int): Nombre de champions à retourner (par défaut 3)
    
    Returns:
        List[str]: Liste des noms des champions triés par maîtrise
    """
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')  # Ne pas ouvrir la fenêtre Chrome (optionnel)
    driver = webdriver.Chrome(service=Service(), options=options)
    # Aller sur la page
    url = f"https://xdx.gg/{player_id}-{region}"
    driver.get(url)

    # Attendre que la page charge (ajuster si nécessaire)
    time.sleep(5)

    # Trouver les divs avec les champions filtrés
    champ_divs = driver.find_elements(By.CSS_SELECTOR, 'div.Filter_champfilteractive__uPUFa')

    # Extraire les données
    champions = []
    for div in champ_divs:
        data_id = div.get_attribute('data-id')
        img_tag = div.find_element(By.TAG_NAME, 'img')
        champ_name = img_tag.get_attribute('alt') if img_tag else "Unknown"
        champions.append({champ_name: data_id})

    #print(f"champions : {champions}")

    top_champs = [list(champ.keys())[0] for champ in champions[:top_n]]

    #print(f"top five : {top_champs}")
    # Fermer le navigateur
    driver.quit()
    return top_champs

def recommend_enemy_best_team(players_champions):
    """
    Recommande la meilleure composition ennemie basée sur les winrates des champions,
    en assignant à chaque joueur un de ses champions maîtrisés.
    
    Args:
        players_champions: Dictionnaire {player_id: [champion1, champion2, ...]}
        
    Returns:
        Dictionnaire {player_id: best_champion} avec un champion par joueur
    """
    if not players_champions:
        return {}
    
    # Convertir les données en format exploitable
    players = list(players_champions.keys())
    champion_lists = list(players_champions.values())
    
    # Générer toutes les combinaisons possibles
    from itertools import product
    all_combinations = product(*champion_lists)
    
    best_composition = {}
    best_avg_winrate = 0
    
    # Évaluer chaque combinaison possible
    for combination in all_combinations:
        current_composition = dict(zip(players, combination))
        
        # Calculer le winrate moyen de cette composition
        total_winrate = 0
        valid = True
        
        for champ in combination:
            champ_data = champion_winrates_df[champion_winrates_df["Champion"] == champ]
            if not champ_data.empty:
                total_winrate += champ_data.iloc[0]["Predicted_Winrate"]
            else:
                valid = False
                break
        
        if not valid:
            continue
            
        avg_winrate = total_winrate / len(combination)
        
        # Conserver la meilleure composition
        if avg_winrate > best_avg_winrate:
            best_avg_winrate = avg_winrate
            best_composition = current_composition
    
    # Si aucune combinaison valide n'a été trouvée, prendre simplement le premier champion de chaque
    if not best_composition:
        best_composition = {player: champs[0] for player, champs in players_champions.items() if champs}
    
    return best_composition


