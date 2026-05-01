import os

PROJECT = "autoresearch_mlops"

FILES = {

f"{PROJECT}/program.md": """# AUTORESEARCH PROGRAM

GOAL:
Minimize RMSE for regression.

METRIC:
RMSE < 0.1

MODELS:
- nn
- xgboost

STOP:
- RMSE < 0.1
- no improvement 10 trials
- max 50 trials
""",

f"{PROJECT}/main.py": """from data import load_data
from core import run

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()
    run(X_train, X_test, y_train, y_test)
""",

f"{PROJECT}/config.py": """TARGET_RMSE = 0.1
MAX_TRIALS = 50
MAX_PATIENCE = 10
""",

f"{PROJECT}/data.py": """import pandas as pd
from sklearn.model_selection import train_test_split

def load_data():
    df = pd.read_csv("data.csv")
    X = df.drop("target", axis=1)
    y = df["target"]
    return train_test_split(X, y, test_size=0.2)
""",

f"{PROJECT}/models.py": """import torch
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
""",

f"{PROJECT}/eval.py": """import numpy as np
import torch

def evaluate(model, X, y, model_type):
    if model_type == "xgb":
        pred = model.predict(X)
    else:
        with torch.no_grad():
            pred = model(torch.tensor(X, dtype=torch.float32)).numpy()

    return np.sqrt(((pred.flatten() - y) ** 2).mean())
""",

f"{PROJECT}/memory.py": """import numpy as np

class Memory:
    def __init__(self):
        self.data = []

    def add(self, r):
        self.data.append(r)

    def recent(self, k=5):
        return self.data[-k:]
""",

f"{PROJECT}/tracker.py": """import mlflow

class Tracker:
    def __init__(self):
        mlflow.set_experiment("autoresearch")

    def log(self, config, metrics):
        with mlflow.start_run():
            mlflow.log_params(config)
            mlflow.log_metrics(metrics)
""",

f"{PROJECT}/llm.py": """import requests

def suggest(history, program):
    prompt = f\"\"\"
{program}

History:
{history}

Return JSON config.
\"\"\"

    r = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "gemma:2b",
            "prompt": prompt,
            "stream": False
        }
    )

    try:
        return eval(r.json()["response"])
    except:
        return {}
""",

f"{PROJECT}/core.py": """import optuna
from models import NN, train_xgb
from eval import evaluate
from memory import Memory
from tracker import Tracker
from llm import suggest

memory = Memory()
tracker = Tracker()

best = float("inf")
patience = 0

def run(X_train, X_test, y_train, y_test):

    def objective(trial):
        global best, patience

        config = {
            "model_type": trial.suggest_categorical("model", ["nn","xgb"]),
            "lr": trial.suggest_float("lr", 1e-4, 1e-1, log=True),
            "epochs": trial.suggest_int("epochs", 10, 50)
        }

        program = open("program.md").read()
        llm_cfg = suggest(memory.recent(), program)
        config.update({k:v for k,v in llm_cfg.items() if k in config})

        if config["model_type"] == "xgb":
            model = train_xgb(X_train, y_train, config)
        else:
            model = NN(X_train.shape[1])

        rmse = evaluate(model, X_test, y_test, config["model_type"])

        tracker.log(config, {"rmse": rmse})
        memory.add({**config, "rmse": rmse})

        if rmse < best:
            best = rmse
            patience = 0
        else:
            patience += 1

        if best < 0.1 or patience > 10:
            trial.study.stop()

        return rmse

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50)
"""
}

def create():
    for path, content in FILES.items():
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

    print("✅ AutoResearch project created successfully!")

if __name__ == "__main__":
    create()