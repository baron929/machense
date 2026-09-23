import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from src.model_training import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    get_mlflow_registry_config,
    load_data,
    load_model_bundle,
    load_validated_model,
    register_model_in_registry,
    resolve_model_reference,
    split_data,
    train_model,
    validate_model_feature_contract,
)
from src.object_storage import get_object_storage_config, get_s3_client, upload_file


def ai4i_rows(size=10):
    return {
        "Type": ["L", "M"] * (size // 2),
        "Air temperature [K]": list(range(298, 298 + size)),
        "Process temperature [K]": list(range(308, 308 + size)),
        "Rotational speed [rpm]": list(range(1400, 1400 + size)),
        "Torque [Nm]": list(range(40, 40 + size)),
        "Tool wear [min]": list(range(size)),
        "Machine failure": [0, 1] * (size // 2),
    }


def test_split_data_uses_ai4i_features_and_stratifies_rows():
    data = pd.DataFrame(ai4i_rows())

    X_train, X_validation, y_train, y_validation = split_data(
        data, validation_fraction=0.2
    )

    assert X_train.columns.tolist() == FEATURE_COLUMNS
    assert len(X_train) == 8
    assert len(X_validation) == 2
    assert y_train.sum() == 4
    assert y_validation.sum() == 1


def test_load_data_requires_ai4i_target(tmp_path):
    path = tmp_path / "maintenance.csv"
    pd.DataFrame({"temperature": [10]}).to_csv(path, index=False)

    with pytest.raises(ValueError, match="Machine failure"):
        load_data(str(path))


def test_load_data_rejects_missing_ai4i_features(tmp_path):
    path = tmp_path / "maintenance.csv"
    data = ai4i_rows()
    data.pop("Torque [Nm]")
    pd.DataFrame(data).to_csv(path, index=False)

    with pytest.raises(ValueError, match=r"Torque \[Nm\]"):
        load_data(str(path))


def test_train_model_saves_schema_versioned_bundle(tmp_path):
    data_path = tmp_path / "ai4i.csv"
    pd.DataFrame(ai4i_rows(40)).to_csv(data_path, index=False)
    artifact_path = tmp_path / "artifacts" / "models" / "latest"

    result = train_model(str(data_path), model_path=str(artifact_path))

    assert Path(result["bundle_path"]).exists()
    loaded_bundle = load_model_bundle(Path(result["bundle_path"]))
    assert loaded_bundle["schema_version"] == "1.0"
    assert loaded_bundle["features"] == FEATURE_COLUMNS
    assert loaded_bundle["target"] == TARGET_COLUMN
    assert loaded_bundle["seed"] == 42
    assert loaded_bundle["model_version"] == result["model_version"]
    assert set(loaded_bundle["metrics"]).issuperset({"accuracy", "precision", "recall"})
    assert loaded_bundle["dataset_sha256"]
    assert hasattr(loaded_bundle["model"], "predict")
    mlmodel_metadata = (artifact_path / "MLmodel").read_text(encoding="utf-8")
    assert "signature:" in mlmodel_metadata


def test_load_model_bundle_rejects_feature_mismatch(tmp_path):
    data_path = tmp_path / "ai4i.csv"
    pd.DataFrame(ai4i_rows(40)).to_csv(data_path, index=False)

    bundle_path = tmp_path / "bundle.json"
    bundle = {
        "schema_version": "1.0",
        "model": None,
        "features": ["wrong", "feature"],
        "target": TARGET_COLUMN,
        "model_version": "1.0.0",
        "seed": 42,
        "metrics": {"accuracy": 1.0},
        "dataset_sha256": "abc",
    }
    bundle_path.write_text(__import__("json").dumps(bundle), encoding="utf-8")

    with pytest.raises(ValueError, match="feature"):
        load_model_bundle(bundle_path)


def test_load_model_bundle_rejects_payload_checksum_mismatch(tmp_path):
    data_path = tmp_path / "ai4i.csv"
    pd.DataFrame(ai4i_rows(40)).to_csv(data_path, index=False)
    model_path = tmp_path / "model.joblib"
    model_path.write_bytes(b"tampered")
    bundle_path = tmp_path / "bundle.json"
    bundle = {
        "schema_version": "1.0",
        "model": {"path": model_path.name},
        "features": FEATURE_COLUMNS,
        "target": TARGET_COLUMN,
        "model_version": "1.0.0",
        "model_family": "Pipeline",
        "artifact_checksum_sha256": "0" * 64,
    }
    bundle_path.write_text(__import__("json").dumps(bundle), encoding="utf-8")

    with pytest.raises(ValueError, match="checksum mismatch"):
        load_model_bundle(bundle_path)


def test_external_checksum_controls_bundle_verification(tmp_path):
    model_path = tmp_path / "model.joblib"
    model_payload = b"delivered-payload"
    model_path.write_bytes(model_payload)
    bundle_path = tmp_path / "bundle.json"
    bundle = {
        "schema_version": "1.0",
        "model": {"path": model_path.name},
        "features": FEATURE_COLUMNS,
        "target": TARGET_COLUMN,
        "model_version": "1.0.0",
        "model_family": "Pipeline",
        "artifact_checksum_sha256": hashlib.sha256(model_payload).hexdigest(),
    }
    bundle_path.write_text(json.dumps(bundle), encoding="utf-8")

    with pytest.raises(ValueError, match="checksum mismatch"):
        load_model_bundle(bundle_path, expected_checksum="0" * 64)


def test_registry_alias_and_version_must_match(monkeypatch):
    class RegistryVersion:
        name = "MachSenseFailureClassifier"

    class FakeClient:
        def get_model_version(self, name, version):
            result = RegistryVersion()
            result.version = version
            return result

        def get_model_version_by_alias(self, name, alias):
            result = RegistryVersion()
            result.version = "8"
            return result

    monkeypatch.setenv("MODEL_SOURCE", "registry")
    monkeypatch.setenv("MODEL_NAME", "MachSenseFailureClassifier")
    monkeypatch.setenv("MODEL_VERSION", "7")
    monkeypatch.setenv("MODEL_ALIAS", "staging")
    monkeypatch.setattr("mlflow.MlflowClient", FakeClient)

    with pytest.raises(ValueError, match="resolves to version '8'"):
        load_validated_model()


def test_mlflow_registry_config_uses_environment_overrides(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "sqlite:///custom_mlflow.db")
    monkeypatch.setenv("MLFLOW_REGISTRY_NAME", "MachSenseFailureClassifier")
    monkeypatch.setenv("MLFLOW_REGISTRY_ALIAS", "staging")

    config = get_mlflow_registry_config()

    assert config["tracking_uri"] == "sqlite:///custom_mlflow.db"
    assert config["registry_name"] == "MachSenseFailureClassifier"
    assert config["registry_alias"] == "staging"


def test_registry_registration_failure_is_non_blocking(monkeypatch):
    class DummyModel:
        def __init__(self):
            self.n_features_in_ = 6

    def raise_error(*args, **kwargs):
        raise RuntimeError("registry unavailable")

    monkeypatch.setattr("mlflow.register_model", raise_error)

    result = register_model_in_registry(
        model=DummyModel(),
        model_name="MachSenseFailureClassifier",
        run_id="test-run-123",
        artifact_path="model",
    )

    assert result["registered"] is False
    assert result["registry_version"] is None
    assert result["status"] == "skipped"


def test_resolve_model_reference_uses_local_path(monkeypatch):
    monkeypatch.setenv("MODEL_SOURCE", "local")
    monkeypatch.setenv("MODEL_PATH", "custom/local/model")

    source, reference = resolve_model_reference()

    assert source == "local"
    assert reference == "custom/local/model"


def test_resolve_model_reference_requires_registry_version_or_alias(monkeypatch):
    monkeypatch.setenv("MODEL_SOURCE", "registry")
    monkeypatch.delenv("MODEL_VERSION", raising=False)
    monkeypatch.delenv("MODEL_ALIAS", raising=False)
    monkeypatch.setenv("MODEL_NAME", "MachSenseFailureClassifier")

    with pytest.raises(
        ValueError, match="MODEL_VERSION must be set to an explicit non-latest version"
    ):
        resolve_model_reference()


def test_validate_model_feature_contract_rejects_mismatch():
    class DummyModel:
        feature_names_in_ = ["wrong", "columns"]

    with pytest.raises(ValueError, match="feature"):
        validate_model_feature_contract(DummyModel())


def test_filebase_storage_requires_runtime_credentials(monkeypatch):
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
    monkeypatch.setenv("FILEBASE_S3_BUCKET", "failure-sys")

    with pytest.raises(ValueError, match="AWS_ACCESS_KEY_ID"):
        get_object_storage_config()


def test_filebase_s3_client_uses_endpoint_without_logging_secrets(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test-access")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test-secret")
    monkeypatch.setenv("FILEBASE_S3_ENDPOINT", "https://s3.filebase.com")
    monkeypatch.setenv("FILEBASE_S3_REGION", "us-east-1")
    monkeypatch.setenv("FILEBASE_S3_BUCKET", "failure-sys")
    calls = []

    class FakeClient:
        pass

    def fake_client(service, **kwargs):
        calls.append((service, kwargs))
        return FakeClient()

    monkeypatch.setattr("boto3.client", fake_client)

    assert isinstance(get_s3_client(), FakeClient)
    assert calls == [
        (
            "s3",
            {
                "endpoint_url": "https://s3.filebase.com",
                "region_name": "us-east-1",
                "aws_access_key_id": "test-access",
                "aws_secret_access_key": "test-secret",
            },
        )
    ]


def test_filebase_upload_uses_configured_bucket(monkeypatch, tmp_path):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test-access")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test-secret")
    monkeypatch.setenv("FILEBASE_S3_BUCKET", "failure-sys")
    local_path = tmp_path / "artifact.bin"
    local_path.write_bytes(b"artifact")
    calls = []

    class FakeClient:
        def upload_file(self, *args):
            calls.append(args)

    monkeypatch.setattr("src.object_storage.get_s3_client", lambda: FakeClient())

    upload_file(local_path, "models/artifact.bin")

    assert calls == [(str(local_path), "failure-sys", "models/artifact.bin")]
