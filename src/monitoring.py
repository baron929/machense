from evidently.metrics import DataDriftTable
from evidently.report import Report


def generate_drift_report(current_data, reference_data):
    report = Report(metrics=[DataDriftTable()])
    report.run(current_data=current_data, reference_data=reference_data)
    return report
