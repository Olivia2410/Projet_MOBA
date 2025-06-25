from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse,RedirectResponse 
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import pandas as pd
import numpy as np
import os
from typing import Optional
from collections import defaultdict
from analytic import (
    champion_winrates_df, get_pair_synergy,get_counter_score, recommend_champions, analyze_compositions,recommend_best_team,evaluate_all_compositions,get_champion_pick_rates,get_player_mastery_champions,recommend_enemy_best_team,get_champion_roles
)


app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Charger les données
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "static")
champion_winrates_df = pd.read_csv(os.path.join(DATA_DIR, "predicted_champion_winrates.csv"))
champion_roles_df = pd.read_csv(os.path.join(DATA_DIR, "champion_roles.csv"))
lanes_df = pd.read_csv(os.path.join(DATA_DIR, "champion_lanes_summary.csv"))
lanes_df["lanes"] = lanes_df["lanes"].str.split(", ")


@app.get("/", include_in_schema=False)
def redirect_to_champions(request: Request):
    return RedirectResponse(url="/champions")

@app.get("/champions", response_class=HTMLResponse)
def show_champions(request: Request, q: str = "", sort: str = "winrate", order: str = "desc"):
    df = champion_winrates_df.copy()

    # Filtrage par recherche
    if q:
        df = df[df["Champion"].str.contains(q, case=False, na=False)]

    # Récupération des pick rates
    pick_rates = get_champion_pick_rates()

    # Récupération des lanes
    lanes_df = pd.read_csv("static/champion_lanes_summary.csv")
    lanes_df["champion"] = lanes_df["champion"].str.strip()
    lanes_df["lanes"] = lanes_df["lanes"].fillna("")

    # Fusion des données
    df = df.merge(lanes_df, how="left", left_on="Champion", right_on="champion")

    # Construction de la liste finale
    champions = []
    for _, row in df.iterrows():
        champ = row["Champion"]
        winrate = row.get("Predicted_Winrate", 0)
        pick_rate = pick_rates.get(champ, 0)
        lanes = row.get("lanes", "")
        first_lane = lanes.split(',')[0].strip() if lanes else ""
        champions.append((champ, winrate, pick_rate,first_lane))

    # Tri dynamique
    reverse = order == "desc"
    if sort == "winrate":
        champions.sort(key=lambda x: x[1], reverse=reverse)
    elif sort == "pickrate":
        champions.sort(key=lambda x: x[2], reverse=reverse)
    elif sort == "champion":
        champions.sort(key=lambda x: x[0].lower(), reverse=reverse)  # ordre alphabétique

    return templates.TemplateResponse("champions.html", {
        "request": request,
        "champions": champions,
        "query": q,
        "sort": sort,
        "order": order,
        "title": "Champions"
    })


@app.get("/synergy", response_class=HTMLResponse)
async def synergy_page(request: Request):
    champions = sorted(champion_winrates_df["Champion"].tolist())
    champion_roles_df["Champion_roles"] = champion_roles_df["champion"] + " - " + champion_roles_df["roles"]
    champions_roles = dict(zip(champion_roles_df["champion"], champion_roles_df["roles"]))
    return templates.TemplateResponse("synergy.html", {"request": request, "champions": champions,"champions_roles":champions_roles})

@app.post("/synergy", response_class=HTMLResponse)
async def compute_synergy(request: Request, champ1: str = Form(...), champ2: str = Form(...)):
    score = get_pair_synergy(champ1, champ2)
    df = champion_winrates_df.copy()
    pick_rates = get_champion_pick_rates()
    return templates.TemplateResponse("synergy.html", {
        "request": request,
        "result": round(score, 4) if score else "Aucune donnée",
        "champ1": champ1,
        "champ2": champ2,
        "champions": sorted(champion_winrates_df["Champion"].tolist()),
        "champions_roles":dict(zip(champion_roles_df["champion"], champion_roles_df["roles"])),
        "winrates":dict(zip(df["Champion"], df["Predicted_Winrate"])),
        "pick_rates": pick_rates
    })

@app.get("/recommend", response_class=HTMLResponse)
async def draft_page(request: Request):
    champions = sorted(champion_winrates_df["Champion"].tolist())
    return templates.TemplateResponse("recommend.html", {
        "request": request,
        "champions": champions
    })


@app.post("/recommend", response_class=HTMLResponse)
async def recommend(
    request: Request,
    team: str = Form(...),
    enemy: str = Form(...),
    role: Optional[str] = Form(None)
):
    try:
        team_compo = [c.strip() for c in team.split(",") if c.strip()]
        enemy_compo = [c.strip() for c in enemy.split(",") if c.strip()]

        # Appeler la recommandation
        recos = recommend_champions(team_compo, enemy_compo, top_n=5, role=role)

        # Analyse approfondie des compositions
        analysis = analyze_compositions(team_compo, enemy_compo)

        return templates.TemplateResponse("recommend.html", {
            "request": request,
            "team": team,
            "enemy": enemy,
            "role": role,
            "recos": recos,
            "analysis": analysis,
            "champions": sorted(champion_winrates_df["Champion"].tolist())
        })

    except Exception as e:
        return templates.TemplateResponse("recommend.html", {
            "request": request,
            "error": f"Erreur : {str(e)}",
            "champions": sorted(champion_winrates_df["Champion"].tolist())
        })


@app.get("/counter", response_class=HTMLResponse)
async def counter_page(request: Request):
    champions = sorted(champion_winrates_df["Champion"].tolist())
    score = None  # Pas encore de score au premier chargement
    return templates.TemplateResponse("counter.html", {"request": request, "champions": champions, "score": score})


@app.post("/counter", response_class=HTMLResponse)
async def compute_counter_score(request: Request, champ1: str = Form(...), champ2: str = Form(...), depth: int = Form(1)):
    champions = sorted(champion_winrates_df["Champion"].tolist())
    score = get_counter_score(champ1, champ2)
    pick_rates = get_champion_pick_rates()
    df = champion_winrates_df.copy()

    return templates.TemplateResponse("counter.html", {
        "request": request,
        "champions": champions,
        "champ1": champ1,
        "champ2": champ2,
        "score": score,
        "champions_roles":dict(zip(champion_roles_df["champion"], champion_roles_df["roles"])),
        "winrates":dict(zip(df["Champion"], df["Predicted_Winrate"])),
        "pick_rates": pick_rates
    })

@app.get("/team_composition", response_class=HTMLResponse)
async def team_composition_page(request: Request):
    champions = sorted(champion_winrates_df["Champion"].tolist())
    return templates.TemplateResponse("team_composition.html", {
        "request": request,
        "champions": champions,
        "title": "Composition d'équipe"
    })

@app.get("/team_builder", response_class=HTMLResponse)
async def team_builder_page(request: Request):
    # Initialiser les structures de données vides si nécessaire
    my_team = []
    enemy_team = []
    
    # Créer un dictionnaire des champions par rôle
    champions_by_role = {
        "top": [],
        "jungle": [],
        "mid": [],
        "adc": [],
        "support": []
    }

    for _, row in lanes_df.iterrows():
        champion = row["champion"]
        if isinstance(row["lanes"], list):
            for lane in row["lanes"]:
                lane_lower = lane.lower()
                if lane_lower in champions_by_role:
                    champions_by_role[lane_lower].append(champion)

    # Trier les listes de champions
    for role in champions_by_role:
        champions_by_role[role].sort()

    return templates.TemplateResponse("team_builder.html", {
        "request": request,
        "champions_by_role": champions_by_role,
        "my_team": my_team,
        "enemy_team": enemy_team,
        "title": "Team Builder"
    })
@app.post("/team_builder", response_class=HTMLResponse)
async def analyze_team_builder(request: Request):
    form_data = await request.form()
    
    # Initialiser les structures de données
    my_team = []
    enemy_team = []
    analysis_result = None
    analysis_type = form_data.get("analysis_type", "score_global")
    # Traitement des actions
    action = form_data.get("action", "")
    
    # Récupérer l'état actuel des équipes
    # Ma Team
    i = 0
    while f"my_team_role_{i}" in form_data:
        role = form_data[f"my_team_role_{i}"]
        current_champs = form_data.get(f"my_team_champions_{i}", "")
        champs_list = [c.strip() for c in current_champs.split(",") if c.strip()] if current_champs else []
        
        my_team.append({
            "role": role,
            "champion": champs_list[0] if champs_list else "",
            "champions": champs_list
        })
        i += 1
    
    # Team Adverse
    i = 0
    while f"enemy_team_role_{i}" in form_data or f"enemy_player_id_{i}" in form_data:
        role = form_data.get(f"enemy_team_role_{i}", "")
        current_champs = form_data.get(f"enemy_team_champions_{i}", "")
        champs_list = [c.strip() for c in current_champs.split(",") if c.strip()] if current_champs else []
        player_id = form_data.get(f"enemy_player_id_{i}", "")
        region = form_data.get(f"enemy_player_region_{i}", "")
        
        # Vérifier si on a déjà un mode enregistré pour ce joueur
        mode = "normal"
        if any(p.get("index") == i for p in enemy_team):
            mode = next(p["mode"] for p in enemy_team if p.get("index") == i)
        
        enemy_team.append({
            "index": i,  # Pour garder trace de l'index original
            "role": role,
            "player_id": player_id,
            "region": region,
            "champions": champs_list,
            "champion": champs_list[0] if champs_list else "",
            "mode": mode
        })
        i += 1
    
    # Gérer les actions
    if action.startswith("fetch_enemy_champions"):
        # Récupérer les champions maîtrisés pour chaque joueur ennemi
        for player in enemy_team:
            if player.get("player_id") and player.get("region"):
                champions = get_player_mastery_champions(player["player_id"], player["region"])
                player["champions"] = champions
    elif action.startswith("switch_to_id_mode_"):
        # Passer en mode saisie d'ID
        player_index = int(action.split("_")[-1])
        if player_index < len(enemy_team):
            enemy_team[player_index]["mode"] = "id"
    
    elif action.startswith("switch_to_normal_mode_"):
        # Revenir en mode normal
        player_index = int(action.split("_")[-1])
        if player_index < len(enemy_team):
            enemy_team[player_index]["mode"] = "normal"

    elif action.startswith("recommend_enemy_team"):
        # Préparer les données d'entrée
        players_champions = {
            i+1: player["champions"] 
            for i, player in enumerate(enemy_team) 
            if player.get("champions")
        }
        print(f"players_champions:{players_champions}")
        # Obtenir la recommandation
        best_assignments = recommend_enemy_best_team(players_champions)
        print(f"best_assignments:{best_assignments}")
        # Appliquer les résultats
        for player in enemy_team:
            player["champions"] = []
            player["champion"] = ""
        
        for player_idx, champ in best_assignments.items():
            if player_idx - 1 < len(enemy_team):
                enemy_team[player_idx - 1]["champions"] = [champ]
                enemy_team[player_idx - 1]["champion"] = champ
                
                # Déterminer le rôle automatiquement
                champ_roles = get_champion_roles(champ)
                if champ_roles:
                    enemy_team[player_idx - 1]["role"] = champ_roles[0]  # Prend le premier rôle disponible



    elif action.startswith("add_my_player"):
        # Ajouter un nouveau joueur à ma team avec des valeurs par défaut
        my_team.append({
            "role": "top",
            "champion": "",
            "champions": []
        })
    elif action.startswith("remove_my_player_"):
        # Supprimer un joueur de ma team (sauf le premier)
        print(action)
        index = int(action.split("_")[-1])-1
        print(index)
        if index < len(my_team):
            my_team.pop(index)
    
    elif action.startswith("add_enemy_player"):
        # Ajouter un nouveau joueur à l'équipe adverse
        enemy_team.append({
            "role": "top",
            "champion": "",
            "player_id": "",
            "champions": []
        })
    elif action.startswith("remove_enemy_player_"):
        # Supprimer un joueur de l'équipe adverse
        index = int(action.split("_")[-1])
        if index < len(enemy_team):
            enemy_team.pop(index)
    
    elif action.startswith("add_my_champion_"):
        # Ajouter un champion à un joueur de ma team
        player_index = int(action.split("_")[-1])
        new_champ = form_data.get(f"new_my_champion_{player_index}", "").strip()
        
        if new_champ and player_index < len(my_team):
            if "champions" not in my_team[player_index]:
                my_team[player_index]["champions"] = []
            
            if new_champ not in my_team[player_index]["champions"]:
                my_team[player_index]["champions"].append(new_champ)

    elif action.startswith("add_enemy_champion_"):
        # Ajouter un champion à un joueur de l'équipe adverse
        player_index = int(action.split("_")[-1])
        new_champ = form_data.get(f"new_enemy_champion_{player_index}", "").strip()
        
        if new_champ and player_index < len(enemy_team):
            if "champions" not in enemy_team[player_index]:
                enemy_team[player_index]["champions"] = []
            
            if new_champ not in enemy_team[player_index]["champions"]:
                enemy_team[player_index]["champions"].append(new_champ)
    
    elif action.startswith("remove_my_champion_"):
        parts = action.split("_")
        print(parts)
        player_index = int(parts[3])  # Index du joueur (0-based)
        champ = "_".join(parts[4:]).replace("_", " ")  # Nom du champion
        
        # Debug logs
        print(f"Debug - Action: {action}")
        print(f"Debug - Parts: {parts}")
        print(f"Debug - Player index: {player_index}, Team length: {len(my_team)}")
        print(f"Debug - Champion to remove: {champ}")
        
        if player_index < len(my_team):
            if "champions" in my_team[player_index] and champ in my_team[player_index]["champions"]:
                my_team[player_index]["champions"].remove(champ)
                print(f"Debug - Removed {champ} from player {player_index}")
            else:
                print(f"Debug - Champion {champ} not found in player's list")
        else:
            print(f"Error: Player index {player_index} is invalid (max: {len(my_team)-1})")
                
    elif action.startswith("remove_enemy_champion_"):
        parts = action.split("_")
        player_index = int(parts[3])
        champ = "_".join(parts[4:])  # Reconstruire le nom du champion
        champ = champ.replace("_", " ").replace("_", " ")  # Remplacer les underscores par des espaces
        if player_index < len(enemy_team):
            if "champions" in enemy_team[player_index] and champ in enemy_team[player_index]["champions"]:
                enemy_team[player_index]["champions"].remove(champ)
                print(f"Debug - Removed {champ} from player {player_index}")
            else:
                print(f"Debug - Champion {champ} not found in player's list")
        else:
            print(f"Error: Player index {player_index} is invalid (max: {len(enemy_team)-1})")

    # Créer le dictionnaire champions_by_role
    champions_by_role = {
        "top": [],
        "jungle": [],
        "mid": [],
        "adc": [],
        "support": []
    }
    for _, row in lanes_df.iterrows():
        champion = row["champion"]
        if isinstance(row["lanes"], list):
            for lane in row["lanes"]:
                lane_lower = lane.lower()
                if lane_lower in champions_by_role:
                    champions_by_role[lane_lower].append(champion)
    # Trier les listes de champions
    for role in champions_by_role:
        champions_by_role[role].sort()

    # Si l'utilisateur a demandé une analyse
    if action == "analyze":
        # Extraire les champions sélectionnés
        my_champions = []
        for player in my_team:
            my_champions.extend(player["champions"])
        
        enemy_champions = []
        for player in enemy_team:
            enemy_champions.extend(player["champions"])
        
        # Convertir en format pour analytic2.py
        player_champions = defaultdict(list)
        for player in my_team:
            if player["role"] and player["champions"]:
                player_champions[player["role"]] = player["champions"]
        
        # Effectuer l'analyse selon le type sélectionné
        if analysis_type == "meilleur_winrate":
            best = recommend_best_team(player_champions, enemy_champions if enemy_champions else None, "winrate")
        elif analysis_type == "meilleur_synergie":
            best = recommend_best_team(player_champions, enemy_champions if enemy_champions else None, "synergy")
        elif analysis_type == "meilleur_counter":
            best = recommend_best_team(player_champions, enemy_champions if enemy_champions else None, "counter")
        else:  # score_global
            best = recommend_best_team(player_champions, enemy_champions if enemy_champions else None, "overall")
        
        # Préparer les résultats pour l'affichage
        if best:
            analysis_result = {
                "recommended_composition": dict(zip(best["composition"].keys(), best["composition"].values())),
                "win_rate": best.get("avg_winrate", 0),
                "synergy_score": best.get("avg_synergy", 0),
                "counter_score": best.get("counter_advantage", 0),
                "recommendations": [best.get("criteria", "Analyse complète")],
                "alternative_champions": {}  # Vous pouvez remplir ceci si nécessaire
            }
        else:
            analysis_result = None
    
    # S'assurer qu'il y a au moins un joueur dans ma team
    if not my_team:
        my_team.append({
            "role": "top",
            "champion": "",
            "champions": []
        })
    
    return templates.TemplateResponse("team_builder.html", {
        "request": request,
        "analysis_result": analysis_result,
        "analysis_type": analysis_type,
        "my_team": my_team,
        "enemy_team": enemy_team,
        "champions_by_role": champions_by_role,
        "title": "Team Builder"
    })
if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
