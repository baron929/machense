import asyncio

import httpx
import pytest

from deployment import main


class FakeModel:
    def predict(self, values):
        assert len(values) == 1
        assert values.iloc[0]["Type"] == "M"
        return [1]

    def predict_proba(self, values):
        return [[0.125, 0.875]]


def request(method, path, **kwargs):
    async def send_request():
        async with main.app.router.lifespan_context(main.app):
            transport = httpx.ASGITransport(app=main.app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                return await client.request(method, path, **kwargs)

    return asyncio.run(send_request())


def test_health_and_readiness(monkeypatch):
    monkeypatch.setattr(main, "model", FakeModel())
    monkeypatch.setattr(main, "load_model", lambda: None)
    response = request("GET", "/health")
    assert response.json() == {"status": "healthy"}
    assert request("GET", "/ready").status_code == 200


def test_load_model_resolves_registry_reference_explicitly(monkeypatch):
    monkeypatch.setenv("MODEL_SOURCE", "registry")
    monkeypatch.setenv("MODEL_NAME", "MachSenseFailureClassifier")
    monkeypatch.setenv("MODEL_VERSION", "7")

    calls = []

    def fake_resolve_model_reference():
        calls.append("resolve")
        return "registry", "models:/MachSenseFailureClassifier/7"

    def fake_load_validated_model(model_reference=None):
        calls.append(model_reference)
        return FakeModel()

    monkeypatch.setattr(main, "resolve_model_reference", fake_resolve_model_reference)
    monkeypatch.setattr(main, "load_validated_model", fake_load_validated_model)

    main.model = None
    main.model_load_error = None
    main.load_model()

    assert main.model is not None
    assert calls == ["resolve", None]


def test_lifespan_starts_with_validated_model(monkeypatch):
    loaded = FakeModel()
    monkeypatch.setattr(main, "resolve_model_reference", lambda: ("local", "bundle"))
    monkeypatch.setattr(main, "load_validated_model", lambda reference: loaded)
    main.model = None

    async def start_and_stop():
        async with main.app.router.lifespan_context(main.app):
            assert main.model is loaded

    asyncio.run(start_and_stop())


def test_lifespan_fails_with_invalid_artifact(monkeypatch, tmp_path):
    monkeypatch.setenv("MODEL_SOURCE", "local")
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "invalid-model"))
    main.model = FakeModel()

    async def start_and_stop():
        async with main.app.router.lifespan_context(main.app):
            pass

    with pytest.raises(FileNotFoundError, match="does not exist"):
        asyncio.run(start_and_stop())
    assert main.model is None


def test_prediction_returns_model_metadata(monkeypatch):
    monkeypatch.setattr(main, "model", FakeModel())
    monkeypatch.setattr(main, "load_model", lambda: None)
    response = request(
        "POST",
        "/predict",
        json={
            "Type": "M",
            "Air temperature [K]": 298.1,
            "Process temperature [K]": 308.6,
            "Rotational speed [rpm]": 1551,
            "Torque [Nm]": 42.8,
            "Tool wear [min]": 0,
        },
        headers={"X-Request-ID": "test-request"},
    )

    assert response.status_code == 200
    assert response.json()["predicted_failure"] is True
    assert response.json()["failure_probability"] == 0.875
    assert response.json()["request_id"] == "test-request"
    assert response.headers["X-Request-ID"] == "test-request"
    assert "inference_time_ms" in response.json()


def test_prediction_rejects_non_finite_sensor_values(monkeypatch):
    monkeypatch.setattr(main, "model", FakeModel())
    monkeypatch.setattr(main, "load_model", lambda: None)
    response = request(
        "POST",
        "/predict",
        json={
            "Type": "M",
            "Air temperature [K]": "NaN",
            "Process temperature [K]": 308.6,
            "Rotational speed [rpm]": 1551,
            "Torque [Nm]": 42.8,
            "Tool wear [min]": 0,
        },
    )

    assert response.status_code == 422


def test_prediction_does_not_expose_internal_errors(monkeypatch):
    class BrokenModel:
        def predict(self, values):
            raise RuntimeError("secret internal path")

    monkeypatch.setattr(main, "model", BrokenModel())
    monkeypatch.setattr(main, "load_model", lambda: None)
    response = request(
        "POST",
        "/predict",
        json={
            "Type": "M",
            "Air temperature [K]": 298.1,
            "Process temperature [K]": 308.6,
            "Rotational speed [rpm]": 1551,
            "Torque [Nm]": 42.8,
            "Tool wear [min]": 0,
        },
    )

    assert response.status_code == 500
    assert response.json() == {"detail": "prediction failed"}
