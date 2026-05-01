import numpy as np
import torch

def evaluate(model, X, y, model_type):

    # ✅ ALWAYS convert DataFrame → numpy first
    if hasattr(X, "values"):
        X = X.values
    if hasattr(y, "values"):
        y = y.values

    if model_type == "xgb":
        pred = model.predict(X)

    else:
        X_tensor = torch.tensor(X, dtype=torch.float32)

        with torch.no_grad():
            pred = model(X_tensor).numpy()

    rmse = np.sqrt(((pred.flatten() - y) ** 2).mean())
    return rmse