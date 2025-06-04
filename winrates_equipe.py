import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder

# ==== 1. Chargement des données ====

df = pd.read_csv("top_match_data.csv")

blue_cols = ['blue_top', 'blue_jungle', 'blue_mid', 'blue_adc', 'blue_support']
red_cols = ['red_top', 'red_jungle', 'red_mid', 'red_adc', 'red_support']

# Encodage des noms de champions
all_champions = pd.concat([df[blue_cols], df[red_cols]], axis=1).values.ravel()
label_encoder = LabelEncoder()
label_encoder.fit(all_champions)

df[blue_cols] = df[blue_cols].apply(label_encoder.transform)
df[red_cols] = df[red_cols].apply(label_encoder.transform)

# Construction des données
edges = []
targets = []

for _, row in df.iterrows():
    red_team = row[red_cols].values.tolist()
    blue_team = row[blue_cols].values.tolist()
    target = 1.0 if row["winner"] == "Red" else 0.0

    edges.append((red_team, blue_team))
    targets.append(target)

num_champions = len(label_encoder.classes_)

# ==== 2. Dataset et modèle ====

class MatchDataset(Dataset):
    def __init__(self, edges, targets):
        self.edges = edges
        self.targets = targets

    def __len__(self):
        return len(self.edges)

    def __getitem__(self, idx):
        red, blue = self.edges[idx]
        target = self.targets[idx]
        return torch.tensor(red), torch.tensor(blue), torch.tensor(target, dtype=torch.float32)

class ProbabilisticAdditiveModel(nn.Module):
    def __init__(self, num_champions):
        super().__init__()
        self.strengths = nn.Parameter(torch.randn(num_champions))

    def forward(self, red, blue):
        red_strength = self.strengths[red].sum(dim=1)
        blue_strength = self.strengths[blue].sum(dim=1)
        return torch.sigmoid(red_strength - blue_strength)

# ==== 3. Entraînement ====

def train(model, dataloader, epochs=30, lr=0.05):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()

    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0

        for red, blue, target in dataloader:
            prob = model(red, blue)
            loss = loss_fn(prob, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            predictions = (prob > 0.5).float()
            correct += (predictions == target).sum().item()
            total += len(target)

        acc = correct / total
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}, Accuracy: {acc:.2%}")

# ==== 4. Lancement entraînement ====

dataset = MatchDataset(edges, targets)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

model = ProbabilisticAdditiveModel(num_champions)
train(model, dataloader, epochs=30, lr=0.05)

# ==== 5. Prédiction des probabilités ====

model.eval()
probas = []
teams = []

# Fonction inverse pour récupérer les noms des champions
def inverse_transform_teams(teams, label_encoder):
    return [label_encoder.inverse_transform(team) for team in teams]

with torch.no_grad():
    for red, blue in edges:
        red_tensor = torch.tensor([red])
        blue_tensor = torch.tensor([blue])
        prob = model(red_tensor, blue_tensor).item()
        probas.append(prob)

        # Inverser les IDs en noms
        red_names = label_encoder.inverse_transform(red)
        blue_names = label_encoder.inverse_transform(blue)
        
        teams.append({
            'red_team': red_names.tolist(),
            'blue_team': blue_names.tolist(),
            'red_win_prob': prob,
            'blue_win_prob': 1 - prob
        })

# ==== 6. Sauvegarde dans un nouveau fichier ====

# Créer un DataFrame avec les résultats
result_df = pd.DataFrame(teams)

# Ajouter les autres colonnes originales au DataFrame
result_df['game_date'] = df['game_date']
result_df['game_duration'] = df['game_duration']
result_df['winner'] = df['winner']

# Sauvegarde dans un fichier CSV
result_df.to_csv("match_predictions_with_names.csv", index=False)
print("✅ Fichier 'match_predictions_with_names.csv' généré avec les probabilités de victoire et les noms des champions.")

def predict_red_win(red_team_ids, blue_team_ids):
    model.eval()
    with torch.no_grad():
        red_tensor = torch.tensor([red_team_ids])
        blue_tensor = torch.tensor([blue_team_ids])
        prob = model(red_tensor, blue_tensor).item()
        print(f"✅ Proba que l'équipe rouge gagne : {prob * 100:.1f}%")
        return prob
