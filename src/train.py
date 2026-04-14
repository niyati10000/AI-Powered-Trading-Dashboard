import torch
import pandas as pd
from src.model import SimpleModel

def train():
    df = pd.read_csv("data/processed/final_dataset.csv")

    features = df[['sentiment', 'fear']].values
    targets = df['target'].values
    magnitude = df['pct_change'].fillna(0).values

    X = torch.tensor(features).float()
    y = torch.tensor(targets).long()
    m = torch.tensor(magnitude).float()

    model = SimpleModel()

    loss_dir = torch.nn.CrossEntropyLoss()
    loss_mag = torch.nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(15):
        dir_out, mag_out = model(X)

        loss1 = loss_dir(dir_out, y)
        loss2 = loss_mag(mag_out.squeeze(), m)

        loss = loss1 + loss2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch}, Loss: {loss.item()}")

    torch.save(model.state_dict(), "models/attention_lstm.pth")

if __name__ == "__main__":
    train()