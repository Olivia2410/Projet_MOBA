import requests
import pandas as pd
import os

def get_latest_version():
    url = "https://ddragon.leagueoflegends.com/api/versions.json"
    versions = requests.get(url).json()
    return versions[0]

def fetch_champion_roles(save_csv=True):
    version = get_latest_version()
    url = f"https://ddragon.leagueoflegends.com/cdn/{version}/data/fr_FR/champion.json"
    response = requests.get(url)
    data = response.json()

    roles = []
    for champ_data in data["data"].values():
        name = champ_data["id"]
        tags = champ_data["tags"]  # ["Fighter", "Assassin"], etc.
        roles.append({
            "champion": name,
            "roles": ",".join(tags).lower()
        })

    df = pd.DataFrame(roles)
    
    if save_csv:
        df.to_csv("champion_roles.csv", index=False)
        print("✅ champion_roles.csv généré automatiquement")
    return df

# === À exécuter une fois au lancement de l'appli ===
def ensure_roles_file():
    if not os.path.exists("champion_roles.csv"):
        print("📥 Téléchargement des rôles de champions depuis Riot...")
        fetch_champion_roles()
    else:
        print("✅ Fichier des rôles déjà présent.")

if __name__ == "__main__":
    ensure_roles_file()
