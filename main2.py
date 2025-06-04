from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import pandas as pd
import numpy as np
import os
from analytic2 import (
    champion_winrates_df, get_pair_synergy,get_counter_score, recommend_champions, analyze_compositions,recommend_best_team,evaluate_all_compositions,get_champion_pick_rates 
)


app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Charger les données
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "static")
champion_winrates_df = pd.read_csv(os.path.join(DATA_DIR, "predicted_champion_winrates.csv"))
champion_roles_df = pd.read_csv(os.path.join(DATA_DIR, "champion_roles.csv"))

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("base.html", {"request": request, "title": "Accueil"})

@app.get("/champions", response_class=HTMLResponse)
def show_champions(request: Request, q: str = ""):
    df = champion_winrates_df.copy()
    if q:
        df = df[df["Champion"].str.contains(q, case=False)]

    # Récupérer les taux de sélection
    pick_rates = get_champion_pick_rates()
    
    # Créer une liste de tuples (champion, winrate, pick_rate)
    champions = []
    for _, row in df.iterrows():
        champ = row["Champion"]
        champions.append((
            champ,
            row["Predicted_Winrate"],
            pick_rates.get(champ, 0)
        ))
    
    return templates.TemplateResponse("champions.html", {
        "request": request,
        "champions": champions,
        "query": q,
        "title": "Champions"
    })
"""
@app.get("/champions", response_class=HTMLResponse)
def show_champions(request: Request, q: str = ""):
    df = champion_winrates_df.copy()
    if q:
        df = df[df["Champion"].str.contains(q, case=False)]

    champions = list(zip(df["Champion"], df["Predicted_Winrate"]))
    return templates.TemplateResponse("champions.html", {
        "request": request,
        "champions": champions,
        "query": q,
        "title": "Champions"
    })
"""
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
    return templates.TemplateResponse("synergy.html", {
        "request": request,
        "result": round(score, 4) if score else "Aucune donnée",
        "champ1": champ1,
        "champ2": champ2,
        "champions": sorted(champion_winrates_df["Champion"].tolist()),
        "champions_roles":dict(zip(champion_roles_df["champion"], champion_roles_df["roles"])),
        "winrates":dict(zip(df["Champion"], df["Predicted_Winrate"]))
    })
#A ameliorer on doit mettre obligé le mm nb de champions pour l'équipe et l'équipe adverse
@app.get("/recommend", response_class=HTMLResponse)
async def draft_page(request: Request):
    champions = sorted(champion_winrates_df["Champion"].tolist())
    return templates.TemplateResponse("recommend.html", {"request": request, "champions": champions})

@app.post("/recommend", response_class=HTMLResponse)
async def recommend(
    request: Request,
    team: str = Form(...),
    enemy: str = Form(...),
):
    team_compo = [c.strip() for c in team.split(",") if c.strip()]
    enemy_compo = [c.strip() for c in enemy.split(",") if c.strip()]
    recos = recommend_champions(team_compo, enemy_compo, top_n=5)
    analysis = analyze_compositions(team_compo, enemy_compo)

    return templates.TemplateResponse("recommend.html", {
        "request": request,
        "team": team,
        "enemy": enemy,
        "recos": recos,
        "analysis": analysis,
        "champions": sorted(champion_winrates_df["Champion"].tolist())
    })


@app.get("/counter", response_class=HTMLResponse)
async def counter_page(request: Request):
    champions = sorted(champion_winrates_df["Champion"].tolist())
    return templates.TemplateResponse("counter.html", {"request": request, "champions": champions})


@app.post("/counter", response_class=HTMLResponse)
async def compute_counter_score(request: Request, champ1: str = Form(...), champ2: str = Form(...), depth: int = Form(1)):
    champions = sorted(champion_winrates_df["Champion"].tolist())
    score = get_counter_score(champ1, champ2, depth=depth)

    return templates.TemplateResponse("counter.html", {
        "request": request,
        "champions": champions,
        "champ1": champ1,
        "champ2": champ2,
        "score": score,
        "depth": depth
    })

@app.get("/team_composition", response_class=HTMLResponse)
async def team_composition_page(request: Request):
    champions = sorted(champion_winrates_df["Champion"].tolist())
    return templates.TemplateResponse("team_composition.html", {
        "request": request,
        "champions": champions,
        "title": "Composition d'équipe"
    })

@app.post("/team_composition", response_class=HTMLResponse)
async def analyze_team_composition(
    request: Request,
    top_champs: str = Form(...),
    jungle_champs: str = Form(...),
    mid_champs: str = Form(...),
    enemy_compo: str = Form(""),
    strategy: str = Form("overall")
):
    # Préparer les données des joueurs
    player_champions = {
        "top": [c.strip() for c in top_champs.split(",") if c.strip()],
        "jungle": [c.strip() for c in jungle_champs.split(",") if c.strip()],
        "mid": [c.strip() for c in mid_champs.split(",") if c.strip()]
    }
    
    # Préparer la composition ennemie si fournie
    enemy = [c.strip() for c in enemy_compo.split(",") if c.strip()] if enemy_compo else None
    
    # Obtenir la recommandation
    recommendation = recommend_best_team(player_champions, enemy, strategy)
    
    # Évaluer toutes les compositions pour les afficher
    all_compositions = evaluate_all_compositions(player_champions, enemy)
    
    return templates.TemplateResponse("team_composition.html", {
        "request": request,
        "recommendation": recommendation,
        "all_compositions": all_compositions,
        "player_champions": player_champions,
        "enemy_compo": enemy_compo,
        "strategy": strategy,
        "champions": sorted(champion_winrates_df["Champion"].tolist()),
        "title": "Résultats de composition"
    })

if __name__ == "__main__":
    uvicorn.run("main2:app", host="127.0.0.1", port=8000, reload=True)