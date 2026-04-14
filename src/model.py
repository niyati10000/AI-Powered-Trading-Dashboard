import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(2, 32)
        self.relu = nn.ReLU()

        self.fc_dir = nn.Linear(32, 2)   # UP/DOWN
        self.fc_mag = nn.Linear(32, 1)   # % change

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)

        direction = self.fc_dir(x)
        magnitude = self.fc_mag(x)

        return direction, magnitude