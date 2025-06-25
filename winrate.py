import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict,Counter
import matplotlib.pyplot as plt
import numpy as np
import os


# 1. Charger les données et calculer les winrates réels
df = pd.read_csv("static/top_match_data.csv")
played = defaultdict(int)
wins = defaultdict(int)

pair_played = defaultdict(int)
pair_wins = defaultdict(int)


for _, row in df.iterrows():
    blue_team = [row['blue_top'], row['blue_jungle'], row['blue_mid'], row['blue_adc'], row['blue_support']]
    red_team = [row['red_top'], row['red_jungle'], row['red_mid'], row['red_adc'], row['red_support']]
    for champ in blue_team + red_team:
        played[champ] += 1
    if row['winner'] == 'Blue':
        for champ in blue_team:
            wins[champ] += 1
    else:
        for champ in red_team:
            wins[champ] += 1

    

champions = sorted(list(played.keys()))
champion_to_idx = {champ: i for i, champ in enumerate(champions)}

# 2. Création des tenseurs pour l'entraînement
true_winrates = torch.tensor(
    [wins[champ] / played[champ] for champ in champions],
    dtype=torch.float32
)

# 3. Déclaration d’un paramètre appris par champion
# Ce sera notre prédiction
pred_winrates = nn.Parameter(torch.full_like(true_winrates, 0.5))

# 4. Définition de la fonction de perte et de l’optimiseur
criterion = nn.MSELoss()
optimizer = optim.SGD([pred_winrates], lr=0.01)

# 5. Entraînement
epochs = 1000
for epoch in range(epochs):
    optimizer.zero_grad()
    loss = criterion(pred_winrates, true_winrates)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.6f}")

# 6. Export des résultats
with torch.no_grad():
    predicted_winrates = pred_winrates.numpy()

df_resultats = pd.DataFrame({
    "Champion": champions,
    "True_Winrate": true_winrates.numpy(),
    "Predicted_Winrate": predicted_winrates
})
#df_resultats.to_csv("predicted_champion_winrates.csv", index=False)

#7. Extrait les lanes des champions (top,mid ...)
champion_lane_counts = defaultdict(Counter)

# Remplir les compteurs
for _, row in df.iterrows():
    for lane in ["top", "jungle", "mid", "adc", "support"]:
        champion_lane_counts[row[f"blue_{lane}"]][lane.capitalize()] += 1
        champion_lane_counts[row[f"red_{lane}"]][lane.capitalize()] += 1

# Créer les données formatées
summary_data = []
for champion, lane_counter in sorted(champion_lane_counts.items()):
    # Trier les lanes par fréquence décroissante
    sorted_lanes = sorted(lane_counter.items(), key=lambda x: x[1], reverse=True)
    # Format : "Lane (count)"
    formatted_lanes = [f"{lane}" for lane, count in sorted_lanes]
    summary_data.append({
        "champion": champion,
        "lanes": ", ".join(formatted_lanes)
    })

# Créer le DataFrame final
summary_df = pd.DataFrame(summary_data)

# Créer le dossier static s’il n’existe pas
os.makedirs("static", exist_ok=True)

# Enregistrer dans static/
output_path = os.path.join("static", "champion_lanes_summary.csv")
summary_df.to_csv(output_path, index=False)

print(f"CSV généré dans : {output_path}")
output_path = os.path.join("static", "predicted_champion_winrates.csv")
df_resultats.to_csv(output_path, index=False)
print(f"Fichier généré avec succès : {output_path}")


champion_indices = np.arange(len(champions))

plt.figure(figsize=(12, 6))
plt.scatter(champion_indices, true_winrates.numpy(), label="Winrate réel", color="blue", s=60)
plt.plot(champion_indices, predicted_winrates, label="Winrate prédit", color="red", linewidth=2)

# Ajouter les labels des champions en abscisse si tu veux
plt.xticks(ticks=champion_indices, labels=champions, rotation=90)

plt.title("Comparaison des winrates réels vs prédits par champion")
plt.xlabel("Champions")
plt.ylabel("Winrate")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()