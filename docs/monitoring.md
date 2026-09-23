# Monitoring

## Current signals

- `/health`: process health.
- `/ready`: model readiness and model version.
- `/metrics`: request, prediction, and error counters with no machine ID or request ID labels.
- Structured API logs include model-load and prediction-failure events.
- `src/monitoring.py` generates an Evidently drift report from reference and current data.

## Interpretation

Operational metrics identify availability, error, and throughput problems. Drift identifies distribution changes. Neither proves model correctness. Ground-truth performance, false alarms per machine, and lead-time analysis require labeled production outcomes and are not implemented.

## Future monitoring

Add latency histograms, validation-failure counters, model-performance reports, and alert rules only after their names, retention, and cardinality are defined. Grafana and Alertmanager are deferred until a real Prometheus deployment exists.