import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict
from itertools import combinations
import numpy as np
import matplotlib.pyplot as plt
import os

"""Dans cette partie la prédiction est 'truqué' car notre base est assez limité ce qui entraine a des winrates 'anormals' on a des
valeurs 1 et 0 ce qui est presque impossible, mais cela est liée au fait qu'on se base sur un nombre de match trop faible qui ne permet pas 
d'avoir des winrates significatifs  """
# --- 1. Charger les données ---
df = pd.read_csv("static/top_match_data.csv")
roles = ['top', 'jungle', 'mid', 'adc', 'support']

# --- 2. Préparer les paires et leurs winrates ---
pair_played = defaultdict(int)
pair_wins = defaultdict(int)
champions_set = set()

for _, row in df.iterrows():
    blue = [row[f'blue_{r}'] for r in roles]
    red = [row[f'red_{r}'] for r in roles]

    for pair in combinations(blue, 2):
        pair_played[tuple(sorted(pair))] += 1
        if row['winner'] == 'Blue':
            pair_wins[tuple(sorted(pair))] += 1
    for pair in combinations(red, 2):
        pair_played[tuple(sorted(pair))] += 1
        if row['winner'] == 'Red':
            pair_wins[tuple(sorted(pair))] += 1

    champions_set.update(blue + red)

# --- 3. Encoder les champions ---
champions = sorted(champions_set)
champion_to_idx = {champ: i for i, champ in enumerate(champions)}
num_champions = len(champions)

# --- 4. Construire le dataset ---
data = []
for (c1, c2), n_played in pair_played.items():
    if n_played < 3:  # optionnel : ignorer les paires rares
        continue
    wr = pair_wins[(c1, c2)] / n_played
    data.append((champion_to_idx[c1], champion_to_idx[c2], wr))

# --- 5. Tensor dataset ---
c1_idx = torch.tensor([d[0] for d in data])
c2_idx = torch.tensor([d[1] for d in data])
labels = torch.tensor([d[2] for d in data], dtype=torch.float32)

# --- 6. Modèle avec embeddings ---
class PairModel(nn.Module):
    def __init__(self, num_champions, emb_dim=8):
        super().__init__()
        self.emb = nn.Embedding(num_champions, emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim * 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Winrate entre 0 et 1
        )

    def forward(self, c1_idx, c2_idx):
        e1 = self.emb(c1_idx)
        e2 = self.emb(c2_idx)
        x = torch.cat([e1, e2], dim=1)
        return self.mlp(x).squeeze()

model = PairModel(num_champions)
optimizer = optim.Adam(model.parameters(), lr=0.001)  # LR plus bas
criterion = nn.BCELoss()  # BCELoss au lieu de MSELoss

# --- 7. Entraînement ---
for epoch in range(100):  # Moins d'epochs
    model.train()
    optimizer.zero_grad()
    output = model(c1_idx, c2_idx)
    loss = criterion(output, labels)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.6f}")

# --- 8. Prédiction et visualisation ---
model.eval()
with torch.no_grad():
    preds = model(c1_idx, c2_idx).numpy()

champion_1 = [champions[i] for i in c1_idx]
champion_2 = [champions[i] for i in c2_idx]

df_result = pd.DataFrame({
    "Champion_1": champion_1,
    "Champion_2": champion_2,
    "True_Winrate": labels.numpy(),
    "Predicted_Winrate": preds
})
output_path = os.path.join("static", "pair_winrate_predictions_with_embeddings.csv")
df_result.to_csv(output_path, index=False)
print("✅ Résultats sauvegardés dans pair_winrate_predictions_with_embeddings.csv")

# --- 9. Plot scatter ---
plt.figure(figsize=(10, 6))
plt.scatter(labels.numpy(), preds, alpha=0.6)
plt.plot([0, 1], [0, 1], 'r--', label="Parfait")
plt.xlabel("Winrate réel")
plt.ylabel("Winrate prédit")
plt.title("Prédiction de winrate des paires avec embeddings")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# --- 10. Histogrammes des distributions ---
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.hist(labels.numpy(), bins=20, alpha=0.7, label="Winrate réel")
plt.legend()
plt.subplot(1,2,2)
plt.hist(preds, bins=20, alpha=0.7, label="Winrate prédit")
plt.legend()
plt.tight_layout()
plt.show()
