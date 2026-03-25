"""
check_threshold.py — Accuracy Gate for the Deploy Job
Reads the MLflow Run ID from model_info.txt, queries final_d_accuracy,
and fails (exit 1) if below 85%.
"""

import os
import sys
import mlflow

THRESHOLD = 99.0  # Temporarily set high to force failure

# ── MLflow Setup ──
tracking_uri = os.environ.get("MLFLOW_TRACKING_URI") or "sqlite:///mlflow.db"
mlflow.set_tracking_uri(tracking_uri)

# ── Read Run ID ──
with open("model_info.txt", "r") as f:
    run_id = f.read().strip()

print(f"Checking Run ID: {run_id}")

# ── Query MLflow ──
client = mlflow.tracking.MlflowClient()
run_data = client.get_run(run_id)
accuracy = run_data.data.metrics.get("final_d_accuracy")

if accuracy is None:
    print("❌ ERROR: 'final_d_accuracy' metric not found for this run.")
    sys.exit(1)

print(f"Accuracy: {accuracy:.2f}%")
print(f"Threshold: {THRESHOLD}%")

if accuracy < THRESHOLD:
    print(f"❌ FAILED: Accuracy {accuracy:.2f}% is below threshold {THRESHOLD}%")
    sys.exit(1)
else:
    print(f"✅ PASSED: Accuracy {accuracy:.2f}% meets threshold {THRESHOLD}%")
    sys.exit(0)
