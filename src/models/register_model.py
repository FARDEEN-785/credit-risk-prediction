import mlflow
from mlflow.tracking import MlflowClient


EXPERIMENT_NAME = "Credit_Risk_MLOps"
MODEL_NAME = "CreditRisk_XGBoost"


def register_best_model():
    client = MlflowClient()

    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    runs = client.search_runs(experiment_ids=[experiment.experiment_id])

    # Select XGBoost run (highest recall)
    best_run = max(
        runs,
        key=lambda run: run.data.metrics.get("recall", 0)
    )

    model_uri = f"runs:/{best_run.info.run_id}/model"

    result = mlflow.register_model(
        model_uri=model_uri,
        name=MODEL_NAME
    )

    print(f"✅ Registered model version: {result.version}")


if __name__ == "__main__":
    register_best_model()
