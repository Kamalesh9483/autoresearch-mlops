import torch
import optuna
from models import NN, train_xgb
from eval import evaluate
from memory import Memory
from tracker import Tracker
from llm import suggest

memory = Memory()
tracker = Tracker()

def run(X_train, X_test, y_train, y_test):

    study_best = {"rmse": float("inf"), "patience": 0}

    def objective(trial):

        config = {
            "model_type": trial.suggest_categorical("model", ["nn","xgb"]),
            "lr": trial.suggest_float("lr", 1e-4, 1e-1, log=True),
            "epochs": trial.suggest_int("epochs", 10, 50)
        }

        program = open("program.md").read()

        # LLM every 5 trials (important fix)
        if trial.number % 5 == 0:
            llm_cfg = suggest(memory.recent(), program)
            config.update({k:v for k,v in llm_cfg.items() if k in config})

        # TRAIN
        if config["model_type"] == "xgb":
            model = train_xgb(X_train, y_train, config)

        else:
            model = NN(X_train.shape[1])

            optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
            loss_fn = torch.nn.MSELoss()

            X_t = torch.tensor(X_train.values, dtype=torch.float32)
            y_t = torch.tensor(y_train.values, dtype=torch.float32).view(-1,1)

            for _ in range(config["epochs"]):
                pred = model(X_t)
                loss = loss_fn(pred, y_t)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # SAFE EVAL
        rmse = evaluate(
            model,
            X_test.values,
            y_test.values,
            config["model_type"]
        )

        # LOGGING
        tracker.log(
            config=config,
            metrics={"rmse": rmse},
            model=model
        )

        memory.add({**config, "rmse": rmse})

        # EARLY STOPPING
        if rmse < study_best["rmse"]:
            study_best["rmse"] = rmse
            study_best["patience"] = 0
        else:
            study_best["patience"] += 1

        if study_best["rmse"] < 0.1 or study_best["patience"] > 10:
            trial.study.stop()

        return rmse

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50)