import os

from prefect import flow, task

from src.model_training import train_model


@task
def train_model_task(data_path: str, model_path: str | None = None) -> dict:
    """Run the canonical trainer inside a Prefect task."""
    return train_model(data_path, model_path=model_path)


@flow
def training_pipeline(data_path: str, model_path: str | None = None) -> dict:
    result = train_model_task(data_path, model_path)
    print(f"Model training completed with ROC AUC: {result['roc_auc']:.4f}")
    return result


if __name__ == "__main__":
    DATA_PATH = os.getenv("DATA_PATH", "data/ai4i2020.csv")
    training_pipeline(DATA_PATH)
