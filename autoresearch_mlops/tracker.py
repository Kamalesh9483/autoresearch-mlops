import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri("sqlite:///mlflow.db")

class Tracker:
    def __init__(self):
        mlflow.set_experiment("autoresearch_mlops")

    def log(self, config, metrics, model=None):
        with mlflow.start_run():
            mlflow.log_params(config)
            mlflow.log_metrics(metrics)

            if model is not None:
                try:
                    mlflow.sklearn.log_model(model, "model")
                except Exception as e:
                    print("Model logging failed:", e)