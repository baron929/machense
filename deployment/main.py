import logging
import math
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.model_training import load_validated_model, resolve_model_reference

logger = logging.getLogger("machsense.api")

MODEL_VERSION = os.getenv("MODEL_VERSION", "ai4i-base1")
MODEL_FEATURE_COUNT = 6
MAX_REQUEST_BYTES = 16 * 1024
model = None
model_load_error = None
metrics = {"requests_total": 0, "predictions_total": 0, "errors_total": 0}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    load_model()
    yield


app = FastAPI(
    title="Predictive Maintenance Model API",
    version=MODEL_VERSION,
    lifespan=lifespan,
)


def load_model() -> None:
    """Load the model once and keep failures out of the request stack trace."""
    global model, model_load_error
    try:
        source, reference = resolve_model_reference()
        if source == "local":
            model = load_validated_model(reference)
            logger.info("model_loaded source=%s path=%s", source, reference)
        else:
            model = load_validated_model()
            logger.info("model_loaded source=%s reference=%s", source, reference)
        model_load_error = None
        logger.info("model_loaded version=%s", MODEL_VERSION)
    except Exception as exc:
        model = None
        model_load_error = type(exc).__name__
        logger.error(
            "model_load_failed source=%s error=%s",
            os.getenv("MODEL_SOURCE", "local"),
            model_load_error,
        )
        raise


@app.middleware("http")
async def request_observability(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))[:128]
    content_length = request.headers.get("content-length")
    if (
        content_length
        and content_length.isdigit()
        and int(content_length) > MAX_REQUEST_BYTES
    ):
        return Response(content="request too large", status_code=413)
    metrics["requests_total"] += 1
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Cache-Control"] = "no-store"
    return response


class PredictionRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    machine_type: Literal["L", "M", "H"] = Field(alias="Type")
    air_temperature_k: float = Field(alias="Air temperature [K]")
    process_temperature_k: float = Field(alias="Process temperature [K]")
    rotational_speed_rpm: float = Field(alias="Rotational speed [rpm]")
    torque_nm: float = Field(alias="Torque [Nm]")
    tool_wear_min: float = Field(alias="Tool wear [min]")

    @field_validator(
        "air_temperature_k",
        "process_temperature_k",
        "rotational_speed_rpm",
        "torque_nm",
        "tool_wear_min",
    )
    @classmethod
    def finite_feature(cls, value: float) -> float:
        if not math.isfinite(value) or not (-1e6 < value < 1e6):
            raise ValueError("sensor values must be finite and within supported bounds")
        return value


class PredictionResponse(BaseModel):
    predicted_failure: bool
    failure_probability: float
    model_version: str
    inference_time_ms: float
    request_id: str


@app.get("/")
def root():
    return {"message": "Predictive Maintenance Model API is running"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/ready")
def ready():
    if model is None:
        raise HTTPException(status_code=503, detail="model is not ready")
    return {"status": "ready", "model_version": MODEL_VERSION}


@app.get("/model/info")
def model_info():
    return {
        "model_version": MODEL_VERSION,
        "loaded": model is not None,
        "feature_count": MODEL_FEATURE_COUNT,
        "target": "Machine failure",
    }


@app.get("/metrics")
def prometheus_metrics() -> Response:
    body = (
        "# TYPE machsense_requests_total counter\n"
        f"machsense_requests_total {metrics['requests_total']}\n"
        "# TYPE machsense_predictions_total counter\n"
        f"machsense_predictions_total {metrics['predictions_total']}\n"
        "# TYPE machsense_errors_total counter\n"
        f"machsense_errors_total {metrics['errors_total']}\n"
    )
    return Response(content=body, media_type="text/plain; version=0.0.4")


@app.post("/predict")
def predict(data: PredictionRequest, request: Request) -> PredictionResponse:
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    if model is None:
        metrics["errors_total"] += 1
        raise HTTPException(status_code=503, detail="model is not ready")

    started = time.perf_counter()
    try:
        input_data = [
            [
                data.machine_type,
                data.air_temperature_k,
                data.process_temperature_k,
                data.rotational_speed_rpm,
                data.torque_nm,
                data.tool_wear_min,
            ]
        ]
        input_frame = pd.DataFrame(
            input_data,
            columns=[
                "Type",
                "Air temperature [K]",
                "Process temperature [K]",
                "Rotational speed [rpm]",
                "Torque [Nm]",
                "Tool wear [min]",
            ],
        )
        prediction = model.predict(input_frame)
        probability = float(model.predict_proba(input_frame)[0][1])
        metrics["predictions_total"] += 1
        return PredictionResponse(
            predicted_failure=bool(prediction[0]),
            failure_probability=round(probability, 6),
            model_version=MODEL_VERSION,
            inference_time_ms=round((time.perf_counter() - started) * 1000, 3),
            request_id=request_id,
        )
    except HTTPException:
        raise
    except Exception:
        metrics["errors_total"] += 1
        logger.exception("prediction_failed request_id=%s", request_id)
        raise HTTPException(status_code=500, detail="prediction failed")
