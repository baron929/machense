# Reproducibility

## Local setup

Use the project requirements files and a supported Python environment. Keep the dataset outside source control when it contains sensitive or proprietary telemetry. Set `DATA_PATH` for training and `MODEL_PATH` for serving.

## Training

```powershell
python -m src.model_training
```

The baseline uses fixed model hyperparameters and random seed 42. MLflow records parameters, the split strategy, classification metrics, and the model artifact. The default local tracking backend is `sqlite:///mlflow.db`.

## Verification

Run `make test` and `make lint`. Build the image from the repository root with `make docker-build`. These commands must pass in a configured environment before a deployment claim is made.

## Known gaps

There is no committed dataset snapshot, lock file, model checksum, model-promotion gate, or verified remote MLflow backend. Reproducibility is therefore limited to code, dependency constraints, and documented training behavior.