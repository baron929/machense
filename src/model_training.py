import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Tuple

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

TARGET_COLUMN = "Machine failure"
TIMESTAMP_COLUMNS = ("timestamp", "datetime", "date", "event_time")
DEFAULT_MODEL_PATH = "models/latest_model"
DEFAULT_MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
DEFAULT_MODEL_VERSION = "ai4i-base1"
FEATURE_COLUMNS = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
NUMERIC_FEATURES = FEATURE_COLUMNS[1:]
CATEGORICAL_FEATURES = [FEATURE_COLUMNS[0]]
SKOPS_TRUSTED_TYPES = [
    "sklearn.compose._column_transformer.ColumnTransformer",
    "sklearn.ensemble._forest.RandomForestClassifier",
    "sklearn.pipeline.Pipeline",
    "sklearn.preprocessing._encoders.OneHotEncoder",
    "sklearn.tree._tree.Tree",
]
MODEL_BUNDLE_SCHEMA_VERSION = "1.0"
SUPPORTED_MODEL_BUNDLE_SCHEMAS = {"1.0", "2.0"}
MODEL_BUNDLE_DIRECTORY = Path("artifacts/models")
DEFAULT_MLFLOW_REGISTRY_NAME = "MachSenseFailureClassifier"
DEFAULT_MLFLOW_REGISTRY_ALIAS = "staging"
DEFAULT_MODEL_SOURCE = "local"
DEFAULT_MODEL_NAME = "MachSenseFailureClassifier"
DEFAULT_MODEL_ALIAS = "staging"


def get_git_commit_sha() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except Exception:
        return None
    return None


def get_mlflow_registry_config() -> dict[str, str | None]:
    return {
        "tracking_uri": os.getenv("MLFLOW_TRACKING_URI", DEFAULT_MLFLOW_TRACKING_URI),
        "registry_name": os.getenv(
            "MLFLOW_REGISTRY_NAME", DEFAULT_MLFLOW_REGISTRY_NAME
        ),
        "registry_alias": os.getenv(
            "MLFLOW_REGISTRY_ALIAS", DEFAULT_MLFLOW_REGISTRY_ALIAS
        ),
    }


def register_model_in_registry(
    model: Any,
    model_name: str,
    run_id: str,
    artifact_path: str = "model",
) -> dict[str, Any]:
    """Register a trained model in MLflow Model Registry when available.

    Registration is intentionally non-blocking; local development continues when the
    tracking server or registry is unavailable.
    """
    try:
        registered_model = mlflow.register_model(
            model_uri=f"runs:/{run_id}/{artifact_path}",
            name=model_name,
        )
        version = str(registered_model.version)
        return {
            "registered": True,
            "registry_version": version,
            "registry_alias": None,
            "run_id": run_id,
            "artifact_source": getattr(registered_model, "source", None),
            "status": "registered_unpromoted",
            "error": None,
        }
    except Exception as exc:
        return {
            "registered": False,
            "registry_version": None,
            "registry_alias": None,
            "run_id": run_id,
            "artifact_source": None,
            "status": "skipped",
            "error": str(exc),
        }


def promote_model_alias(
    model_name: str, model_version: str, alias: str
) -> dict[str, str]:
    """Explicitly promote one exact registry version through an alias."""
    if not model_name or not model_version or not alias:
        raise ValueError("model_name, model_version, and alias are required")
    mlflow.set_tracking_uri(get_mlflow_registry_config()["tracking_uri"])
    registry_client = mlflow.MlflowClient()
    registry_version = registry_client.get_model_version(
        name=model_name,
        version=model_version,
    )
    if registry_version.version != model_version:
        raise ValueError(
            f"Requested model version '{model_version}' but registry resolved to "
            f"'{registry_version.version}'"
        )
    mlflow.set_registered_model_alias(
        name=model_name,
        alias=alias,
        version=model_version,
    )
    return {
        "model_name": model_name,
        "model_version": model_version,
        "alias": alias,
        "run_id": str(getattr(registry_version, "run_id", "")),
        "artifact_source": str(getattr(registry_version, "source", "")),
    }


def compute_dataset_sha256(dataset_path: str | Path | None) -> str | None:
    if dataset_path is None:
        return None

    path = Path(dataset_path)
    if not path.exists() or not path.is_file():
        return None

    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_model_bundle(
    model: Pipeline,
    feature_names: list[str],
    target_name: str,
    model_version: str,
    seed: int,
    metrics: dict[str, float],
    dataset_path: str | Path | None,
    output_dir: str | Path | None = None,
) -> Path:
    artifact_root = Path(output_dir) if output_dir else MODEL_BUNDLE_DIRECTORY
    bundle_dir = artifact_root / model_version
    bundle_dir.mkdir(parents=True, exist_ok=True)

    model_path = bundle_dir / "model.joblib"
    bundle_path = bundle_dir / "bundle.json"

    joblib.dump(model, model_path)

    bundle = {
        "schema_version": MODEL_BUNDLE_SCHEMA_VERSION,
        "model": {"path": model_path.name, "format": "joblib"},
        "features": list(feature_names),
        "input_schema": {
            "columns": list(feature_names),
            "dtypes": {
                "Type": "category",
                "Air temperature [K]": "float",
                "Process temperature [K]": "float",
                "Rotational speed [rpm]": "float",
                "Torque [Nm]": "float",
                "Tool wear [min]": "float",
            },
        },
        "target": target_name,
        "model_family": "RandomForestClassifier",
        "model_version": model_version,
        "seed": seed,
        "metrics": {str(key): float(value) for key, value in metrics.items()},
        "dataset_sha256": compute_dataset_sha256(dataset_path),
        "training_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit_sha": get_git_commit_sha(),
        "classification_threshold": 0.5,
        "python_version": os.sys.version.split()[0],
        "dependency_metadata": {
            "mlflow": getattr(mlflow, "__version__", "unknown"),
            "sklearn": __import__("sklearn").__version__,
            "pandas": getattr(pd, "__version__", "unknown"),
            "joblib": getattr(joblib, "__version__", "unknown"),
        },
        "artifact_checksum_sha256": None,
    }
    bundle_path.write_text(
        json.dumps(bundle, indent=2, sort_keys=True), encoding="utf-8"
    )
    bundle["artifact_checksum_sha256"] = hashlib.sha256(
        model_path.read_bytes()
    ).hexdigest()
    bundle_path.write_text(
        json.dumps(bundle, indent=2, sort_keys=True), encoding="utf-8"
    )
    return bundle_path


def validate_model_feature_contract(
    model: Any, expected_features: list[str] | None = None
) -> None:
    expected = expected_features or FEATURE_COLUMNS
    actual = getattr(model, "feature_names_in_", None)
    if actual is None:
        raise ValueError("Model artifact does not expose feature_names_in_")
    if list(actual) != expected:
        raise ValueError(
            "Model feature contract mismatch: expected "
            f"{expected}, got {list(actual)}"
        )


def validate_model_bundle(bundle: dict[str, Any]) -> None:
    schema_version = bundle.get("schema_version")
    if schema_version not in SUPPORTED_MODEL_BUNDLE_SCHEMAS:
        raise ValueError(
            "Unsupported model bundle schema_version: " f"{schema_version}"
        )
    if not isinstance(bundle.get("features"), list) or not bundle["features"]:
        raise ValueError("Model bundle is missing a valid feature list")
    if bundle["features"] != FEATURE_COLUMNS:
        raise ValueError(
            "Model bundle feature list does not match the expected AI4I contract"
        )
    if bundle.get("target") != TARGET_COLUMN:
        raise ValueError("Model bundle target does not match the AI4I contract")
    if bundle.get("model_family") not in {"RandomForestClassifier", "Pipeline", None}:
        raise ValueError("Model bundle model_family is unsupported")
    checksum = bundle.get("artifact_checksum_sha256")
    if checksum is not None and not isinstance(checksum, str):
        raise ValueError("Artifact checksum must be a sha256 string when present")


def load_model_bundle(
    bundle_path: str | Path,
    expected_checksum: str | None = None,
    require_external_checksum: bool = False,
) -> dict[str, Any]:
    bundle_file = Path(bundle_path)
    if bundle_file.is_dir():
        bundle_file = bundle_file / "bundle.json"

    if not bundle_file.exists():
        raise FileNotFoundError(f"Model bundle does not exist: {bundle_file}")

    with bundle_file.open("r", encoding="utf-8") as source:
        bundle = json.load(source)

    validate_model_bundle(bundle)

    model_path = bundle_file.parent / bundle.get("model", {}).get(
        "path", "model.joblib"
    )
    if not model_path.exists():
        raise FileNotFoundError(f"Model bundle payload is missing: {model_path}")

    embedded_checksum = bundle.get("artifact_checksum_sha256")
    checksum_to_verify = expected_checksum or embedded_checksum
    if require_external_checksum and not expected_checksum:
        raise ValueError(
            "An externally supplied artifact checksum is required for delivery"
        )
    if checksum_to_verify:
        if (
            not isinstance(checksum_to_verify, str)
            or len(checksum_to_verify) != 64
            or any(
                character not in "0123456789abcdefABCDEF"
                for character in checksum_to_verify
            )
        ):
            raise ValueError("Artifact checksum must be a 64-character sha256 value")
        actual_checksum = hashlib.sha256(model_path.read_bytes()).hexdigest()
        if actual_checksum.lower() != checksum_to_verify.lower():
            raise ValueError(
                "Model bundle payload checksum mismatch: "
                f"expected {checksum_to_verify}, got {actual_checksum}"
            )

    bundle["model"] = joblib.load(model_path)
    validate_model_feature_contract(
        bundle["model"], expected_features=bundle["features"]
    )
    return bundle


def save_latest_model(
    model: Pipeline,
    output_path: Path,
    input_example: pd.DataFrame | None = None,
    signature: Any | None = None,
) -> None:
    """Save a model by replacing an existing artifact only after success."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = Path(
        tempfile.mkdtemp(prefix=f".{output_path.name}.", dir=output_path.parent)
    )
    try:
        mlflow.sklearn.save_model(
            model,
            str(temporary_path),
            input_example=input_example,
            signature=signature,
            skops_trusted_types=SKOPS_TRUSTED_TYPES,
        )
        if output_path.is_dir() and not output_path.is_symlink():
            shutil.rmtree(output_path)
        elif output_path.exists():
            output_path.unlink()
        temporary_path.rename(output_path)
    except Exception:
        shutil.rmtree(temporary_path, ignore_errors=True)
        raise


def load_data(path: str) -> pd.DataFrame:
    """Load and validate the UCI AI4I machine-failure dataset."""
    data = pd.read_csv(path)
    if data.empty:
        raise ValueError("The training dataset is empty")
    if TARGET_COLUMN not in data.columns:
        raise ValueError(
            f"The training dataset must contain the AI4I target '{TARGET_COLUMN}'"
        )
    if data[TARGET_COLUMN].isna().any():
        raise ValueError(f"The target column '{TARGET_COLUMN}' contains missing values")
    if not set(data[TARGET_COLUMN].unique()).issubset({0, 1}):
        raise ValueError(
            f"The target column '{TARGET_COLUMN}' must contain only 0 or 1"
        )
    missing_columns = [column for column in FEATURE_COLUMNS if column not in data]
    if missing_columns:
        raise ValueError(
            "The AI4I dataset is missing columns: " + ", ".join(missing_columns)
        )
    if data[FEATURE_COLUMNS].isna().any().any():
        raise ValueError("AI4I feature columns must not contain missing values")
    if not data[CATEGORICAL_FEATURES[0]].isin(["L", "M", "H"]).all():
        raise ValueError("The AI4I 'Type' column must contain only L, M, or H")
    if not all(
        pd.api.types.is_numeric_dtype(data[column]) for column in NUMERIC_FEATURES
    ):
        raise ValueError("AI4I sensor columns must be numeric")
    return data


def split_data(
    data: pd.DataFrame, validation_fraction: float = 0.2
) -> Tuple[pd.DataFrame, ...]:
    """Split AI4I features and target with a reproducible, stratified holdout."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    missing_columns = [
        column for column in [TARGET_COLUMN, *FEATURE_COLUMNS] if column not in data
    ]
    if missing_columns:
        raise ValueError(
            "The AI4I dataset is missing columns: " + ", ".join(missing_columns)
        )

    feature_data = data[FEATURE_COLUMNS].copy()
    target = data[TARGET_COLUMN]
    timestamp_column = next(
        (column for column in TIMESTAMP_COLUMNS if column in data.columns), None
    )

    if timestamp_column:
        timestamps = pd.to_datetime(data[timestamp_column], errors="coerce")
        if timestamps.isna().any():
            raise ValueError(
                f"Timestamp column '{timestamp_column}' contains invalid values"
            )
        ordered = data.assign(_parsed_timestamp=timestamps).sort_values(
            "_parsed_timestamp"
        )
        feature_data = ordered[FEATURE_COLUMNS]
        target = ordered[TARGET_COLUMN]
        split_index = int(len(feature_data) * (1 - validation_fraction))
        if split_index <= 0 or split_index >= len(feature_data):
            raise ValueError(
                "The dataset must contain enough rows for a train/validation split"
            )
        return (
            feature_data.iloc[:split_index],
            feature_data.iloc[split_index:],
            target.iloc[:split_index],
            target.iloc[split_index:],
        )
    else:
        return train_test_split(
            feature_data,
            target,
            test_size=validation_fraction,
            random_state=42,
            stratify=target,
        )


def resolve_model_reference() -> tuple[str, str]:
    model_source = os.getenv("MODEL_SOURCE", DEFAULT_MODEL_SOURCE).strip().lower()
    if model_source == "local":
        model_path = os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH)
        if not model_path:
            raise ValueError("MODEL_PATH must be set when MODEL_SOURCE=local")
        return "local", model_path

    if model_source in {"registry", "mlflow"}:
        model_name = os.getenv("MODEL_NAME", DEFAULT_MODEL_NAME)
        model_version = os.getenv("MODEL_VERSION")
        model_alias = os.getenv("MODEL_ALIAS")
        if not model_name:
            raise ValueError("MODEL_NAME must be set when MODEL_SOURCE=registry")
        if not model_version or model_version.lower() == "latest":
            if model_alias:
                model_version = model_alias
            else:
                raise ValueError(
                    "MODEL_VERSION must be set to an explicit non-latest version "
                    "when MODEL_SOURCE=registry"
                )
        reference = (
            f"models:/{model_name}/{model_version}"
            if model_version
            else f"models:/{model_name}@{model_alias}"
        )
        return "registry", reference

    raise ValueError(f"Unsupported MODEL_SOURCE value: {model_source!r}")


def load_validated_model(model_reference: str | None = None) -> Any:
    if model_reference is None:
        source, reference = resolve_model_reference()
    else:
        source = "local"
        reference = model_reference

    if source == "local":
        path = Path(reference)
        if not path.exists():
            raise FileNotFoundError(f"Configured model path does not exist: {path}")
        if path.is_dir():
            bundle_path = path / "bundle.json"
            if not bundle_path.exists():
                raise FileNotFoundError(
                    "Local model directory is missing bundle.json metadata: " f"{path}"
                )
            expected_checksum = os.getenv("MODEL_ARTIFACT_SHA256")
            require_external_checksum = (
                os.getenv("MODEL_ARTIFACT_CHECKSUM_REQUIRED", "false").lower() == "true"
            )
            bundle = load_model_bundle(
                bundle_path,
                expected_checksum=expected_checksum,
                require_external_checksum=require_external_checksum,
            )
            validate_model_feature_contract(
                bundle["model"], expected_features=bundle["features"]
            )
            return bundle["model"]
        model = mlflow.sklearn.load_model(str(path))
        validate_model_feature_contract(model, expected_features=FEATURE_COLUMNS)
        return model

    if source == "registry":
        model_name = os.getenv("MODEL_NAME")
        requested_version = os.getenv("MODEL_VERSION")
        requested_alias = os.getenv("MODEL_ALIAS")
        if not model_name:
            raise ValueError("MODEL_NAME is required for registry model resolution")
        if requested_version and requested_version.lower() == "latest":
            raise ValueError("MODEL_VERSION must be a specific version, not 'latest'")
        if not requested_version and not requested_alias:
            raise ValueError(
                "MODEL_VERSION or MODEL_ALIAS must be set for registry resolution"
            )
        mlflow.set_tracking_uri(get_mlflow_registry_config()["tracking_uri"])
        registry_client = mlflow.MlflowClient()
        try:
            if requested_version:
                registry_version = registry_client.get_model_version(
                    name=model_name,
                    version=requested_version,
                )
                if requested_alias:
                    alias_version = registry_client.get_model_version_by_alias(
                        name=model_name,
                        alias=requested_alias,
                    )
                    if alias_version.version != registry_version.version:
                        raise ValueError(
                            f"Alias '{requested_alias}' resolves to version "
                            f"'{alias_version.version}', not "
                            f"'{registry_version.version}'"
                        )
            else:
                registry_version = registry_client.get_model_version_by_alias(
                    name=model_name,
                    alias=requested_alias,
                )
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError(
                f"Registry model reference for '{model_name}' was not found or "
                f"is unavailable: {exc}"
            ) from exc

        if registry_version.name != model_name:
            raise ValueError(
                f"Requested model '{model_name}' but registry resolved to "
                f"'{registry_version.name}'"
            )
        if requested_version and registry_version.version != requested_version:
            raise ValueError(
                f"Requested model version '{requested_version}' but registry "
                f"resolved to '{registry_version.version}'"
            )

        model = mlflow.sklearn.load_model(
            f"models:/{model_name}/{registry_version.version}"
        )
        validate_model_feature_contract(model, expected_features=FEATURE_COLUMNS)
        return model

    raise ValueError(f"Unsupported model source resolution: {source}")


def train_model(data_path: str, model_path: str | None = None) -> dict[str, Any]:
    data = load_data(data_path)
    X_train, X_val, y_train, y_val = split_data(data)
    model_path = model_path or os.getenv("MODEL_OUTPUT_PATH", DEFAULT_MODEL_PATH)
    registry_config = get_mlflow_registry_config()
    mlflow.set_tracking_uri(registry_config["tracking_uri"])
    model_version = os.getenv("MODEL_VERSION", DEFAULT_MODEL_VERSION)
    model_name = str(registry_config["registry_name"])

    with mlflow.start_run() as run:
        n_estimators = 100
        max_depth = 10
        model = Pipeline(
            steps=[
                (
                    "preprocessor",
                    ColumnTransformer(
                        transformers=[
                            ("numeric", "passthrough", NUMERIC_FEATURES),
                            (
                                "categorical",
                                OneHotEncoder(handle_unknown="ignore"),
                                CATEGORICAL_FEATURES,
                            ),
                        ]
                    ),
                ),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        class_weight="balanced_subsample",
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        )

        model.fit(X_train, y_train)
        predictions = model.predict(X_val)
        probabilities = model.predict_proba(X_val)[:, 1]
        accuracy = float(accuracy_score(y_val, predictions))
        precision = float(precision_score(y_val, predictions, zero_division=0))
        recall = float(recall_score(y_val, predictions, zero_division=0))
        f1 = float(f1_score(y_val, predictions, zero_division=0))
        roc_auc = float(roc_auc_score(y_val, probabilities))

        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("target_column", TARGET_COLUMN)
        mlflow.log_param("model_version", model_version)
        mlflow.log_param("split_strategy", "stratified_random_or_chronological")
        mlflow.log_param("feature_names", ",".join(FEATURE_COLUMNS))
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc_auc)

        input_example = X_val.iloc[[0]].copy()
        input_example[NUMERIC_FEATURES] = input_example[NUMERIC_FEATURES].astype(
            "float64"
        )
        signature = infer_signature(input_example, model.predict(input_example))
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            input_example=input_example,
            signature=signature,
            skops_trusted_types=SKOPS_TRUSTED_TYPES,
        )

        run_id = run.info.run_id
        registry_result = register_model_in_registry(
            model=model,
            model_name=model_name,
            run_id=run_id,
            artifact_path="model",
        )

    output_path = Path(model_path)
    save_latest_model(
        model,
        output_path,
        input_example=input_example,
        signature=signature,
    )
    bundle_metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
    }
    bundle_path = save_model_bundle(
        model=model,
        feature_names=FEATURE_COLUMNS,
        target_name=TARGET_COLUMN,
        model_version=model_version,
        seed=42,
        metrics=bundle_metrics,
        dataset_path=data_path,
        output_dir=MODEL_BUNDLE_DIRECTORY,
    )
    bundle_metadata = json.loads(bundle_path.read_text(encoding="utf-8"))
    release_identity = {
        "model_name": model_name,
        "registry_version": registry_result.get("registry_version"),
        "registry_run_id": registry_result.get("run_id", run.info.run_id),
        "registry_artifact_source": registry_result.get("artifact_source"),
        "model_version": bundle_metadata["model_version"],
        "bundle_path": str(bundle_path),
        "artifact_checksum_sha256": bundle_metadata.get("artifact_checksum_sha256"),
        "git_commit_sha": bundle_metadata.get("git_commit_sha"),
    }

    result = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "target_column": TARGET_COLUMN,
        "model_version": model_version,
        "feature_names": FEATURE_COLUMNS,
        "model_path": str(output_path),
        "bundle_path": str(bundle_path),
        "dataset_sha256": compute_dataset_sha256(data_path),
        "model": model,
        "mlflow_run_id": run.info.run_id,
        "mlflow_model_version": registry_result.get("registry_version"),
        "mlflow_registry": {
            "name": model_name,
            "registered": registry_result["registered"],
            "status": registry_result["status"],
            "error": registry_result.get("error"),
        },
        "release_identity": release_identity,
    }
    print(f"Validation ROC AUC: {roc_auc:.4f}")
    return result


if __name__ == "__main__":
    DATA_PATH = os.getenv("DATA_PATH", "data/ai4i2020.csv")
    train_model(DATA_PATH)
