# Data Card

## Intended data

The bundled UCI AI4I 2020 dataset contains machine type, air/process temperature, rotational speed, torque, tool wear, and a binary `Machine failure` target. Identifier and failure-mode columns are excluded to reduce leakage.

## Current validation

The loader rejects empty datasets, missing AI4I columns, missing values, unknown machine types, and target values outside `{0, 1}`. Numeric sensor columns are validated before training. The dataset has no time-series machine identifier, so the default split is stratified and reproducible.

## Missing information

The repository does not contain the source dataset, collection process, machine population, sampling frequency, labeling method, prediction horizon, retention policy, or known bias/quality analysis. These must be documented before operational claims are made.

## Privacy and retention

Do not place personal data or secrets in telemetry, logs, metrics, or model artifacts. Define retention and access controls with the data owner before adding durable prediction storage.