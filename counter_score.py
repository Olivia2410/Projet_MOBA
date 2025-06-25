import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import os
# --- 1. Charger les données ---
df = pd.read_csv("static/top_match_data.csv")
roles = ['top', 'jungle', 'mid', 'adc', 'support']

# --- 2. Préparer les matchups et leurs winrates ---
matchup_played = defaultdict(int)
matchup_wins = defaultdict(int)
champions_set = set()

for _, row in df.iterrows():
    blue = [row[f'blue_{r}'] for r in roles]
    red = [row[f'red_{r}'] for r in roles]
    
    # Compter les matchups entre équipes adverses
    for blue_champ in blue:
        for red_champ in red:
            matchup = tuple(sorted((blue_champ, red_champ)))
            matchup_played[matchup] += 1
            if row['winner'] == 'Blue':
                matchup_wins[matchup] += 1
    
    champions_set.update(blue + red)

# --- 3. Encoder les champions ---
champions = sorted(champions_set)
champion_to_idx = {champ: i for i, champ in enumerate(champions)}
num_champions = len(champions)

# --- 4. Construire le dataset ---
data = []
for (c1, c2), n_played in matchup_played.items():
    if n_played < 3:  # ignorer les matchups trop rares
        continue
    wr = matchup_wins[(c1, c2)] / n_played
    # On stocke les deux orientations (c1 vs c2 et c2 vs c1)
    data.append((champion_to_idx[c1], champion_to_idx[c2], wr))
    data.append((champion_to_idx[c2], champion_to_idx[c1], 1 - wr))

# --- 5. Tensor dataset ---
attacker_idx = torch.tensor([d[0] for d in data])
defender_idx = torch.tensor([d[1] for d in data])
labels = torch.tensor([d[2] for d in data], dtype=torch.float32)

# --- 6. Modèle avec embeddings ---
class CounterModel(nn.Module):
    def __init__(self, num_champions, emb_dim=8):
        super().__init__()
        self.attacker_emb = nn.Embedding(num_champions, emb_dim)
        self.defender_emb = nn.Embedding(num_champions, emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim * 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Probabilité de victoire entre 0 et 1
        )
    
    def forward(self, attacker_idx, defender_idx):
        e_att = self.attacker_emb(attacker_idx)
        e_def = self.defender_emb(defender_idx)
        x = torch.cat([e_att, e_def], dim=1)
        return self.mlp(x).squeeze()

model = CounterModel(num_champions)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()

# --- 7. Entraînement ---
for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    output = model(attacker_idx, defender_idx)
    loss = criterion(output, labels)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.6f}")

# --- 8. Prédiction et sauvegarde ---
model.eval()
with torch.no_grad():
    preds = model(attacker_idx, defender_idx).numpy()

attacker_champs = [champions[i] for i in attacker_idx]
defender_champs = [champions[i] for i in defender_idx]

df_result = pd.DataFrame({
    "Attacker": attacker_champs,
    "Defender": defender_champs,
    "True_Winrate": labels.numpy(),
    "Predicted_Winrate": preds
})

# Calcul du score de counter (avantage relatif)
df_result["Counter_Score"] = df_result["Predicted_Winrate"] - 0.5
output_path = os.path.join("static", "counter_score_predictions_with_embeddings.csv")
df_result.to_csv(output_path, index=False)
print("✅ Résultats sauvegardés dans counter_score_predictions_with_embeddings.csv")

# --- 9. Visualisation ---
plt.figure(figsize=(10, 6))
plt.scatter(labels.numpy(), preds, alpha=0.6)
plt.plot([0, 1], [0, 1], 'r--', label="Parfait")
plt.xlabel("Winrate réel")
plt.ylabel("Winrate prédit")
plt.title("Prédiction des matchups avec embeddings")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Histogramme des scores de counter
plt.figure(figsize=(8, 5))
plt.hist(df_result["Counter_Score"], bins=50, alpha=0.7)
plt.xlabel("Score de counter (avantage relatif)")
plt.ylabel("Fréquence")
plt.title("Distribution des scores de counter")
plt.grid(True)
plt.tight_layout()
plt.show()