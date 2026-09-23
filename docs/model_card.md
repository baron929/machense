# Model Card: AI4I Failure Classifier

## Purpose

Estimate remaining useful life for industrial equipment from numeric telemetry features.

## Model

The current baseline is a preprocessing pipeline with one-hot encoding and `sklearn.ensemble.RandomForestClassifier` using 100 trees, maximum depth 10, balanced class weights, and random seed 42. Training logs classification metrics to MLflow.

## Inputs and output

Training expects the bundled AI4I schema with `Machine failure` as the target, `Type` as a categorical feature, and five numeric sensor features. Identifiers and failure-mode indicators are excluded. Serving accepts the same named sensor fields and returns a failure class and probability.

## Evaluation

Validation uses a stratified 20 percent holdout for AI4I. Accuracy, precision, recall, and ROC AUC are reported. No production performance, failure lead time, machine-level false-alarm rate, or business impact is claimed.

## Limitations

The repository contains no dataset or trained artifact. Sensor semantics, prediction horizon, machine grouping, calibration, missing-value policy, and operating thresholds require domain-owner confirmation.