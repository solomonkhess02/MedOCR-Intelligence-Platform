"""
MLflow Setup — Initialize Experiments and Model Registry Stages

Run once after Docker services are up:
  python mlflow_setup/init_experiment.py

Creates:
  - TrOCR-Prescription experiment
  - Donut-Invoice experiment
  - LayoutLMv3-LabReport experiment
  - OMR-Classical experiment
"""

import mlflow
from mlflow.tracking import MlflowClient
import sys
import os
from pathlib import Path

# Allow running from backend/ directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.chdir(Path(__file__).resolve().parent.parent)

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

from app.config import get_settings

settings = get_settings()

EXPERIMENTS = [
    "TrOCR-Prescription",
    "Donut-Invoice",
    "LayoutLMv3-LabReport",
    "OMR-Classical",
    "System-Monitoring",
]


def init_experiments():
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = MlflowClient()

    print(f"Connecting to MLflow at: {settings.mlflow_tracking_uri}")

    for exp_name in EXPERIMENTS:
        existing = client.get_experiment_by_name(exp_name)
        if existing is None:
            exp_id = client.create_experiment(
                exp_name,
                tags={
                    "project": "MedOCR Intelligence Platform",
                    "version": "3.0",
                },
            )
            print(f"  ✓ Created experiment: '{exp_name}' (id={exp_id})")
        else:
            print(f"  · Experiment already exists: '{exp_name}' (id={existing.experiment_id})")

    print("\nMLflow experiments initialized successfully.")
    print(f"View MLflow UI at: {settings.mlflow_tracking_uri}")


if __name__ == "__main__":
    init_experiments()
