FROM python:3.12.14-slim-trixie

WORKDIR /app

COPY deployment/requirements-runtime.txt requirements-runtime.txt

RUN pip install --no-cache-dir --upgrade pip "setuptools>=78.1.1,<81" "msgpack>=1.2.1,<2" \
    && pip install --no-cache-dir -r requirements-runtime.txt

COPY src ./src
COPY deployment ./deployment
COPY artifacts/models/ai4i-base1 ./models/latest_model

ARG MODEL_ARTIFACT_SHA256
ARG MODEL_VERSION=ai4i-base1
ENV MODEL_SOURCE=local \
    MODEL_PATH=/app/models/latest_model \
    MODEL_VERSION=${MODEL_VERSION} \
    MODEL_ARTIFACT_SHA256=${MODEL_ARTIFACT_SHA256} \
    MODEL_ARTIFACT_CHECKSUM_REQUIRED=true \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

LABEL org.opencontainers.image.title="MachSense-AI" \
    org.opencontainers.image.model-version="${MODEL_VERSION}" \
    org.opencontainers.image.model-sha256="${MODEL_ARTIFACT_SHA256}"

RUN if [ -n "$MODEL_ARTIFACT_SHA256" ]; then \
      python -c "import os; from src.model_training import load_model_bundle; load_model_bundle('/app/models/latest_model', expected_checksum=os.environ['MODEL_ARTIFACT_SHA256'], require_external_checksum=True)"; \
    else \
      echo 'Skipping model checksum validation (MODEL_ARTIFACT_SHA256 not provided)'; \
    fi

RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/ready')"

CMD ["uvicorn", "deployment.main:app", "--host", "0.0.0.0", "--port", "8000"]
