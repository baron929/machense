# MachSense AI Architecture Audit

## Audit scope and evidence

This is a Phase 0 audit of the repository as inspected on 2026-09-21. It is based on the files currently present in the workspace, not on claims in reference repositories or in the README. The repository contains 15 visible files and no `pyproject.toml`, `.env.example`, `.dockerignore`, Docker Compose file, Kubernetes manifests, database migrations, model artifact, dataset, or package initializer.

The named reference repositories were reviewed at their public GitHub README level:

- `DiogoRibeiro7/fastapi-ml-platform`: structured FastAPI service for fraud scoring with schemas, services, repositories, authentication, prediction persistence, health/readiness, metrics, and integration tests.
- `DuqueOM/ML-MLOps-Production-Template` (currently presented on GitHub as `DuqueOM/ml-service-template`): a large governed, multi-cloud Kubernetes template with supply-chain security, model promotion, observability, and CI/CD controls.
- `prakashkmr48/MLOps-Production-Pipeline`: FastAPI, Docker, Kubernetes, Prometheus/Grafana, CI/CD, health checks, and example operational tests.

These repositories are references only. Their domains, deployment assumptions, and operational scale are not automatically compatible with MachSense AI.

## Existing architecture

The current repository is a small Python MLOps demonstration with four logical parts:

1. `src/model_training.py` loads a CSV, selects `RUL` as the target, trains a `RandomForestRegressor`, calculates validation RMSE, and logs the run and model to MLflow.
2. `flows/prefect_pipeline.py` wraps data loading/splitting and model training/logging in Prefect tasks and a flow.
3. `deployment/main.py` creates a FastAPI application, loads an MLflow model at module import time, validates five generic numeric features with Pydantic, and serves `POST /predict`.
4. `src/monitoring.py` creates an Evidently data-drift report using `DataDriftTable`.

Deployment is represented by a Python 3.10 slim Dockerfile and an incomplete Terraform configuration referencing local modules that are not present in this repository. CI runs pytest and formatting/lint checks on pushes and pull requests to `main`.

## Existing features

- CSV loading with pandas.
- RUL regression target convention.
- Random forest regression with fixed hyperparameters.
- Random train/validation split with `random_state=42`.
- RMSE calculation.
- MLflow run, parameter, metric, and model logging.
- Prefect task/flow orchestration.
- Evidently drift report generation.
- FastAPI OpenAPI generation through the framework.
- Pydantic request validation for five required floats.
- One-time model loading at module import.
- Docker and Terraform placeholders.
- pytest, Black, Flake8, isort, and pre-commit configuration.
- GitHub Actions test and lint job.

## Existing dependencies

`requirements.txt` pins numpy, pandas, scikit-learn, joblib, MLflow, Evidently, Prefect, FastAPI, Uvicorn, PyArrow, boto3, protobuf, Pydantic 1, pytest, httpx, Black, Flake8, isort, pre-commit, python-dotenv, and requests. `requirements-dev.txt` repeats several runtime and quality dependencies.

There is no dependency lock file, optional dependency grouping, vulnerability scanner configuration, or explicit runtime-only dependency set. The README mentions XGBoost, Prometheus, and model registry behavior, but XGBoost and `prometheus-client` are not dependencies and no corresponding implementation is present.

## Existing API routes

Evidence from `deployment/main.py`:

| Method | Path | Behavior | Protection |
|---|---|---|---|
| GET | `/` | Returns a running-message JSON response | None |
| POST | `/predict` | Predicts RUL from five generic float fields | None |

There are no health, readiness, metrics, model metadata, versioned API, machine, authentication, or error-schema routes.

## Existing model pipeline

The implemented target is continuous RUL regression, not binary failure classification. The training module uses a random split rather than a chronological or machine-aware split. It does not validate required columns, feature types, missing values, ranges, duplicate records, machine identity, timestamps, prediction horizon, or leakage. No feature engineering or training/inference preprocessing pipeline is implemented. No evaluation artifact, baseline comparison, PR-AUC, false-positive analysis, lead-time analysis, or model metadata is produced.

MLflow logging is present, but the code does not configure a tracking backend, explicitly register a model, validate a candidate, promote a version, or provide a reproducible model-loading contract. The serving code expects `models/latest_model`, but training logs to MLflow and does not create that path in the repository.

## Existing deployment approach

- Local API command: `uvicorn deployment.main:app --reload` via the Makefile.
- Docker command: build context is `deployment/`, while the Dockerfile attempts `COPY ../requirements.txt` and `COPY ../src`; Docker build contexts cannot copy files above the context directory, so the documented `docker-build` target is expected to fail from the current layout.
- The Docker image runs as root, has no health check, does not include a model artifact or explicit runtime configuration, and has no `.dockerignore`.
- Terraform references `deployment/terraform/modules/gcs` and `modules/compute`, plus undeclared `var.project_id`; those modules and variable declarations are absent.
- No Compose or Kubernetes deployment is present.

## Existing security controls

Present controls are limited to pinned requirements, Pydantic input typing, pre-commit formatting/lint hooks, and CI lint/test commands. There is no authentication, authorization, secret example, secret scanning, dependency audit, container scan, SBOM, image signing, non-root runtime, CI permission hardening, request-size limit, rate limit, structured audit log, or security documentation.

## Existing testing coverage

The two tests only assert `1 + 1 == 2`; they do not import or exercise training, monitoring, the API, model loading, validation, error handling, or deployment. The current environment could not execute them because `pytest` was not found on PATH. CI intends to execute `pytest tests/`, but that has not been verified locally in this audit.

## Documentation and implementation discrepancies

- README architecture describes ingestion, stream processing, feature store, model registry, Prometheus, alerting, and retraining, none of which is implemented here.
- README describes XGBoost classification and failure prediction, while code trains a random-forest RUL regressor.
- README lists files such as `data_preprocessing.py` and `model_inference.py` that are absent.
- README claims `/metrics`, but the API has no metrics route.
- README claims model registry/versioning, but code only logs an MLflow artifact.
- README claims GCP Cloud Run/GKE and Prefect Cloud deployment, but Terraform is incomplete and no deployment workflow exists.
- README contains outcome and accuracy targets that have no evidence in this repository and should be treated as aspirational until measured.

## Features worth preserving

- RUL prediction as the current domain target, unless the product decision explicitly changes it to failure classification.
- The simple training entry point and Prefect orchestration concept.
- MLflow experiment/artifact logging.
- Evidently as an optional drift-analysis component.
- FastAPI and Pydantic for the serving boundary.
- Docker, CI, pytest, and pre-commit as foundations for incremental hardening.
- The existing repository's small-service scope; it does not justify premature microservices or Kubernetes.

## Features that can be improved

- Replace generic feature names with a documented sensor contract and stable feature ordering.
- Separate API schemas, inference/service logic, configuration, and model loading without changing the domain contract unnecessarily.
- Add `/health` and `/ready`, safe error responses, request IDs, bounded input validation, inference latency logging, and a model metadata response.
- Make model loading explicit and testable, with startup/readiness behavior for missing artifacts.
- Use chronological or machine-aware splits when timestamps/machine IDs exist; document the target and horizon.
- Add meaningful unit and API tests.
- Add structured logs and low-cardinality Prometheus metrics only after defining the metric contract.
- Harden Docker and CI incrementally with non-root execution, a `.dockerignore`, safer permissions, dependency auditing, and secret scanning.
- Reconcile the README with verified behavior and add model/data/reproducibility documentation.

## Features that should be removed or corrected

No existing application feature is safe to delete solely because it is absent from a reference repository. The following documentation/configuration claims should be removed or corrected unless implemented and tested:

- Claims of XGBoost classification, 92% accuracy, downtime reduction, and other unverified outcomes.
- Claims that Prometheus, model registry, feature store, stream processing, alerting, and retraining are implemented.
- References to absent source files and unsupported deployment commands.
- The current Docker build target/layout should be corrected; it should not be silently retained as a working deployment claim.

Potentially unused dependencies such as boto3, joblib, requests, and PyArrow require reference searches and runtime verification before removal. They are not deleted in Phase 0.

## Features that should not be added

- Fraud-specific transaction endpoints, customer concepts, or payment workflows from the FastAPI reference.
- Generic user-management or authentication flows before protected administrative/model operations are identified.
- Database, Redis/RQ, and asynchronous batch-job infrastructure without a demonstrated persistence or throughput requirement.
- Kubernetes, HPA, Grafana, multi-cloud Terraform, service meshes, or complex promotion systems before a tested local container exists.
- Feature-store ownership, stream-processing infrastructure, or a second model registry.
- Agentic automation or unrelated computer-vision, chatbot, e-commerce, or payment features.

## Security weaknesses and operational risks

1. `deployment/main.py` exposes prediction without authentication or authorization; model-management routes do not exist, but future administrative routes must not inherit this default.
2. The API returns `str(e)` in a 500 response, which can expose internal paths, dependency details, or model information.
3. Model loading happens at import time and can prevent health checks or tests from starting when the artifact is absent; there is no readiness distinction.
4. Docker runs as root and has no health check or build-context-safe layout.
5. CI uses mutable major-version action tags and does not set explicit least-privilege permissions or run security scans.
6. Dependencies are pinned but not audited, and runtime/dev dependencies are duplicated.
7. Terraform contains a hardcoded bucket name and incomplete module references; deployment cannot currently be verified.
8. No secret management or environment contract exists. Secrets must not be added to source control while these integrations are designed.
9. Input size, finite-number, range, and unexpected-value handling are incomplete.
10. Prediction and inference telemetry are absent, so failures, drift, latency, and model versions cannot be correlated.
11. Random splitting may leak future or same-machine information in time-series data.

## Recommended implementation order

1. Obtain approval for the scope and decide whether MachSense predicts RUL, failure probability, or both. Correct documentation claims accordingly.
2. Establish a runnable baseline: environment setup, dependency verification, import-safe model loading, Docker build context, and meaningful tests.
3. Define data, feature, target, horizon, model artifact, and model metadata contracts; add validation and leakage-aware evaluation.
4. Refactor the serving boundary minimally: typed sensor request/response schemas, service-level inference, health/readiness, safe errors, and model version reporting.
5. Add structured prediction/inference logging with retention guidance; use a simple sink first and avoid a database unless a requirement justifies it.
6. Add bounded metrics and drift/data-quality checks with tests; do not imply model correctness from operational monitoring.
7. Harden CI and containers: non-root image, `.dockerignore`, dependency/secret scanning, least-privilege workflow permissions, and optional SBOM/image scanning where the toolchain is available.
8. Add model/data/evaluation/reproducibility/security/deployment/monitoring documentation and reconcile the README.
9. Reassess authentication for model-management/admin endpoints and only then add a minimal mechanism if the deployment context requires it.
10. Consider Kubernetes only after Docker, health checks, configuration, and local operational tests pass.

## Proposed files for the next approved implementation phase

Likely modified files: `README.md`, `requirements.txt`, `requirements-dev.txt`, `Makefile`, `.github/workflows/ci-cd.yml`, `deployment/main.py`, `deployment/Dockerfile`, `src/model_training.py`, `src/monitoring.py`, `flows/prefect_pipeline.py`, and the existing tests.

Likely new files: `.dockerignore`, `.env.example`, API/service/schema modules only where the refactor is justified, focused tests, and the documentation files listed in the request. Exact file changes should be confirmed after the target contract and environment are approved.

## Potential breaking changes

- Changing the target from RUL regression to binary failure classification changes model artifacts, metrics, request/response semantics, and clients.
- Replacing `feature_1` through `feature_5` with named sensor fields changes the `/predict` request contract.
- Moving or versioning routes changes existing clients.
- Making startup fail closed when a model is missing changes current import behavior.
- Changing the Docker build context or command changes local deployment instructions.
- Adding authentication to `/predict` may break unauthenticated clients.
- Switching from random to chronological splits can change reported metrics and model promotion decisions.

## Phase 0 decision gate

This audit does not delete or rewrite application features. Approval is requested before implementing the substantive changes above, especially any API contract change, authentication, model-target change, dependency removal, Terraform change, or deployment change.
