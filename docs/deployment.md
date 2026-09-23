# Deployment

## Local API

Install dependencies from the repository root, then run:

```powershell
uvicorn deployment.main:app --reload
```

The API listens on port 8000. `/health` reports process health; `/ready` reports whether the configured MLflow model loaded. Set `MODEL_PATH` and optionally `MODEL_VERSION` through the environment. No model artifact is committed to this repository.

## Docker

Build from the repository root so the Dockerfile can access both `requirements.txt` and source code:

```powershell
python -c "from src.model_training import train_model; train_model('data/ai4i2020.csv')"
$checksum = (Get-FileHash artifacts/models/ai4i-base1/model.joblib -Algorithm SHA256).Hash.ToLower()
docker build --build-arg MODEL_ARTIFACT_SHA256=$checksum -t machsense-api:local -f deployment/Dockerfile .
docker run --rm -p 8000:8000 machsense-api:local
```

The build packages `artifacts/models/ai4i-base1` as `/app/models/latest_model`, validates its bundle and checksum during the image build, and configures local model mode explicitly. The image runs as a non-root user and its Docker health check uses `/ready`, so a missing or invalid model cannot appear healthy. Training, Prefect orchestration, and drift-report dependencies remain outside the production image.

The local bundle is ignored by Git and must be generated or supplied before building a fresh checkout:

The checksum is supplied by the delivery process, not trusted from `bundle.json`. Do not put registry credentials in the Dockerfile. A registry URL is not required for this local-bundle image. It is required only when deploying with `MODEL_SOURCE=registry`, in which case provide `MLFLOW_TRACKING_URI`, `MODEL_NAME`, and either an explicit `MODEL_VERSION` or `MODEL_ALIAS` through the runtime secret/configuration system. When both version and alias are supplied, they must resolve to the same registry version.

Filebase S3-compatible storage is configured with `FILEBASE_S3_ENDPOINT`, `FILEBASE_S3_REGION`, and `FILEBASE_S3_BUCKET`. The application reads `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` from the runtime environment; credentials are never stored in the repository or logged. The S3 client is available through `src.object_storage.get_s3_client()` and the `upload_file`/`download_file` helpers.

## Docker Hub delivery

The repository includes a separate [Docker Hub release workflow](../.github/workflows/dockerhub-release.yml). It runs tests, creates the versioned model bundle, calculates the model checksum, builds the image, validates `/ready` and `/predict`, and pushes only on `main`.

Configure these GitHub values before enabling a production push:

- Repository variable `DOCKERHUB_NAMESPACE`: Docker Hub namespace (for this repository, `aram07`).
- Repository variable `DOCKERHUB_REPOSITORY`: Docker Hub repository name (`machsense-api`).
- Repository secret `DOCKERHUB_USERNAME`: Docker Hub login username.
- Repository secret `DOCKERHUB_TOKEN`: Docker Hub access token with the minimum required push permission.

This resolves to a production image name of `aram07/machsense-api` unless the repository variables are overridden.

The workflow publishes an immutable `sha-<full-commit-sha>` tag. It does not push `latest`, and it records the model version and artifact SHA-256 in the workflow output. No image is pushed from pull requests or without the required configuration.

## Cloud and Kubernetes status

Terraform currently defines only an optional versioned GCS artifact bucket. Kubernetes manifests are intentionally deferred. A cloud deployment should not be attempted until the image, model artifact delivery, readiness behavior, secret handling, and operational tests are verified.