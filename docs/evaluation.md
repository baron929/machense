# Evaluation

## Current procedure

Run training against `data/ai4i2020.csv`. The training module reports validation accuracy, precision, recall, and ROC AUC and logs them to MLflow. If a recognized timestamp is present, validation is the latest 20 percent after chronological ordering; otherwise the split is stratified with a fixed seed.

## Required future evidence

Record the target definition, dataset version, split policy, feature preprocessing, accuracy, precision, recall, ROC AUC, false alarms, and missed failures. A model must not be promoted from a single aggregate score, especially with this imbalanced target.

## Leakage controls

Do not randomly mix future observations into training. Keep training-only preprocessing fitted inside the training partition and verify that machine identity or duplicated records do not cross the split.