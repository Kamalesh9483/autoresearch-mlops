import torch
import torch.nn as nn
from xgboost import XGBRegressor

class NN(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.net(x)

def train_xgb(X, y, cfg):
    model = XGBRegressor(
        learning_rate=cfg["lr"],
        n_estimators=cfg["epochs"]
    )
    model.fit(X, y)
    return model
