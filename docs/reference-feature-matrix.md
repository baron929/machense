# Reference Feature Compatibility Matrix

This matrix compares verified local implementation with relevant patterns from the three named public reference repositories. It is a planning artifact, not authorization to copy their domain code or deployment stack.

| Feature | Existing implementation | Reference source | Relevant to MachSense? | Action | Reason |
|---|---|---|---|---|---|
| Predictive-maintenance RUL target | `RUL` regression in `src/model_training.py` | Primary architecture | Yes | PRESERVE | This is the current domain behavior and should remain the foundation until the target is explicitly changed. |
| Sensor schema and request validation | Five generic required floats in `deployment/main.py` | FastAPI ML Platform | Yes | IMPROVE | Keep typed validation, but use named industrial sensor fields and finite/range checks based on real data. |
| API routers/services/repositories | Single `deployment/main.py`; no repositories | FastAPI ML Platform | Partly | DEFER | Separate route and inference service first; add persistence/repository layers only when prediction logging needs durable storage. |
| Model loading at application startup | MLflow model loaded at module import | FastAPI ML Platform | Yes | IMPROVE | Preserve one-time loading, but use lifespan/state, explicit readiness, and a testable missing-artifact path. |
| Health endpoint | No health endpoint | FastAPI ML Platform; MLOps Production Pipeline | Yes | INTEGRATE | Needed for container and operational checks; implement before Kubernetes. |
| Readiness endpoint | No readiness endpoint | FastAPI ML Platform; MLOps Production Pipeline | Yes | INTEGRATE | Model availability differs from process health. |
| Structured error envelope | Raw exception text returned as HTTP 500 detail | FastAPI ML Platform | Yes | IMPROVE | Prevent internal detail exposure and make client handling consistent. |
| Prediction logging | No prediction logging | FastAPI ML Platform; ML-MLOps Production Template | Yes | INTEGRATE | Add minimal structured records for traceability, model version, status, latency, and retention. |
| Database-backed prediction store | No database | FastAPI ML Platform | Maybe | DEFER | A database adds operational cost; first use structured logs or a simple justified sink. |
| API-key/JWT authentication | None | FastAPI ML Platform; MLOps Production Pipeline | Conditional | DEFER | Protect admin/model operations if added; do not add generic identity infrastructure before endpoint threat modeling. |
| Role-based authorization | None | FastAPI ML Platform | Conditional | DEFER | Only needed when distinct operator/admin/service permissions exist. |
| Rate limiting and request-size limits | None | FastAPI ML Platform | Yes for exposed service | DEFER | Add with deployment context and capacity limits; not required for the first local training baseline. |
| MLflow experiment logging | Present for parameters, RMSE, and model artifact | Primary architecture; ML-MLOps Production Template | Yes | PRESERVE | Existing compatible lifecycle capability. |
| Model registry/promotion | No verified registry or promotion path | README claim; reference templates | Yes | IMPROVE | First define artifact metadata and validation gates; avoid a second registry. |
| Time-aware split and leakage checks | Random `train_test_split` | Primary predictive-maintenance requirements; ML-MLOps Production Template | Yes | INTEGRATE | Industrial telemetry can leak future or machine information through random splitting. |
| Data validation contract | No schema, range, missingness, or leakage checks | ML-MLOps Production Template | Yes | INTEGRATE | Needed for reliable sensor ingestion and training/inference parity. |
| Feature store | Not implemented; README-only claim | Primary README; ML-MLOps Production Template scope boundary | No for current size | REJECT | Unnecessary platform ownership for this repository. |
| Stream processing | Not implemented; README-only claim | Primary README | Not yet | DEFER | Add only with a real streaming source and latency requirement. |
| Evidently drift report | `DataDriftTable` helper exists | FastAPI ML Platform; ML-MLOps Production Template | Yes | PRESERVE | Compatible with data-quality monitoring, but add baseline/versioning and tests. |
| Prometheus metrics | No implementation or dependency | FastAPI ML Platform; MLOps Production Pipeline | Yes | INTEGRATE | Add low-cardinality API/inference metrics after defining names and labels. |
| Grafana dashboards | None | MLOps Production Pipeline; ML-MLOps Production Template | Yes later | DEFER | Dashboards depend on stable metrics and an actual monitoring deployment. |
| Structured application logs | `print` in training/flow; no API logging | FastAPI ML Platform; ML-MLOps Production Template | Yes | INTEGRATE | Needed for request/model/inference troubleshooting without raw payloads. |
| Docker containerization | Dockerfile exists but copy paths are invalid for its build context | MLOps Production Pipeline | Yes | IMPROVE | Fix the build contract and run as non-root with a health check. |
| Multi-stage Docker build | Single stage | MLOps Production Pipeline | Useful | DEFER | Consider after a working image and measured size/build need. |
| Non-root container | Not implemented | MLOps Production Pipeline; ML-MLOps Production Template | Yes | INTEGRATE | Low-cost runtime hardening. |
| `.dockerignore` | Missing | Deployment best practice in references | Yes | INTEGRATE | Prevent accidental source, test, cache, and secret inclusion in build context. |
| Docker Compose | Missing | FastAPI ML Platform; MLOps Production Pipeline | Maybe | DEFER | Add only if local monitoring or service dependencies require it. |
| Kubernetes manifests | Missing | MLOps Production Pipeline; ML-MLOps Production Template | Later | DEFER | Docker and readiness must work before adding cluster complexity. |
| Terraform infrastructure | Incomplete module references and undeclared variable | Primary repository | Conditional | IMPROVE | Repair or document as a separate deployment track; do not claim deployability until validated. |
| CI tests and lint | GitHub Actions runs pytest/Black/Flake8/isort | All references | Yes | IMPROVE | Keep the pipeline, pin actions/permissions, and make tests meaningful. |
| Security scanning | None | ML-MLOps Production Template; MLOps Production Pipeline | Yes | INTEGRATE | Start with secret/dependency scanning and add container/SBOM gates when reproducible. |
| SBOM and image signing | None | ML-MLOps Production Template | Later | DEFER | Valuable for release governance, but premature before a stable image/promotion flow. |
| Model/database batch jobs | None | FastAPI ML Platform | No current evidence | REJECT | Fraud batch-job infrastructure is unrelated to the present narrow serving path. |
| Fraud transaction domain | None | FastAPI ML Platform | No | REJECT | Conflicts with industrial equipment health and anomaly detection. |
| Generic user-management system | None | FastAPI ML Platform | No | REJECT | Add only narrowly scoped operator authorization if required. |
| Kubernetes multi-cloud governance | None | ML-MLOps Production Template | Not initially | REJECT | Scale and governance exceed current repository evidence and stated first-version boundary. |
| Load testing | None | MLOps Production Pipeline | Yes later | DEFER | Add after the API contract and model artifact path are stable. |
| Model cards/data cards/reproducibility docs | Missing | Requested target; ML-MLOps Production Template | Yes | INTEGRATE | Documents target, limitations, data lineage, metrics, and repeatability without runtime complexity. |
| Unverified performance claims | README claims 92% accuracy and business outcomes | Existing README | No as facts | REMOVE | Claims lack dataset, experiment, and production evidence. |

## Proposed next phase

The recommended first implementation slice is: make the existing RUL service truthful and runnable, define the sensor/model contract, add meaningful tests, add health/readiness and safe errors, and repair the Docker build. Authentication, durable prediction storage, Prometheus/Grafana deployment, supply-chain signing, and Kubernetes remain gated follow-up work.
