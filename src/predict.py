import torch
from src.model import SimpleModel

model = SimpleModel()
model.load_state_dict(torch.load("models/attention_lstm.pth"))
model.eval()

def predict(features):
    x = torch.tensor(features).float().unsqueeze(0)

    dir_out, mag_out = model(x)

    prob = torch.softmax(dir_out, dim=1)

    up_prob = prob[0][1].item()
    down_prob = prob[0][0].item()

    movement = "UP" if up_prob > down_prob else "DOWN"
    confidence = max(up_prob, down_prob)

    magnitude = mag_out.item()

    return movement, magnitude, confidence