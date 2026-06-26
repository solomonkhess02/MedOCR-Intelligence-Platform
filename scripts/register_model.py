"""
Register the fine-tuned TrOCR model in the MLflow Model Registry.

This backs the resume claim of "model versioning": it logs the locally fine-tuned
TrOCR model as an MLflow run artifact and registers it under a named registered model,
creating a new version each time it is run. You can then promote versions through
stages (None → Staging → Production) in the MLflow UI or via MlflowClient.

Prereqs:
  - MLflow server running:  python -m mlflow server --host 127.0.0.1 --port 5000
  - A fine-tuned model present at models/trocr-finetuned/

Run:
  $env:PYTHONIOENCODING="utf-8"
  .venv\\Scripts\\python.exe scripts\\register_model.py
"""

import sys
from pathlib import Path

import mlflow
import mlflow.transformers

MLFLOW_URI = "http://127.0.0.1:5000"
EXPERIMENT = "TrOCR-Prescription"
REGISTERED_MODEL_NAME = "TrOCR-Prescription"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FINETUNED_DIR = PROJECT_ROOT / "models" / "trocr-finetuned"


def main() -> int:
    if not (FINETUNED_DIR / "config.json").exists():
        print(f"ERROR: No fine-tuned model found at {FINETUNED_DIR}")
        print("Run the TrOCR fine-tuning (scripts/train_trocr.py --full) first.")
        return 1

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT)
    print(f"Connecting to MLflow at {MLFLOW_URI}, experiment '{EXPERIMENT}'")

    with mlflow.start_run(run_name="register-trocr-finetuned") as run:
        mlflow.log_params({
            "base_model": "microsoft/trocr-base-handwritten",
            "model_version_tag": "trocr-prescription-v2-finetuned",
            "dataset": "medocr-vision-dataset",
            "train_domain": "handwritten_prescriptions",
        })

        registered = False
        # Preferred path: log via the native transformers flavor (loadable as a pipeline).
        try:
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel

            print("Loading fine-tuned model + processor for transformers-flavor logging...")
            model = VisionEncoderDecoderModel.from_pretrained(str(FINETUNED_DIR))
            processor = TrOCRProcessor.from_pretrained(str(FINETUNED_DIR))

            mlflow.transformers.log_model(
                transformers_model={
                    "model": model,
                    "image_processor": processor.image_processor,
                    "tokenizer": processor.tokenizer,
                },
                artifact_path="model",
                task="image-to-text",
                registered_model_name=REGISTERED_MODEL_NAME,
            )
            registered = True
            print("Logged + registered via mlflow.transformers flavor.")
        except Exception as e:
            print(f"transformers-flavor logging failed ({e}); falling back to raw artifacts.")

        # Fallback: log the raw model directory and register it by run URI.
        if not registered:
            mlflow.log_artifacts(str(FINETUNED_DIR), artifact_path="model")
            model_uri = f"runs:/{run.info.run_id}/model"
            mlflow.register_model(model_uri, REGISTERED_MODEL_NAME)
            print(f"Logged raw artifacts and registered {model_uri}.")

        print(f"\nDone. Run ID: {run.info.run_id}")
        print(f"Registered model '{REGISTERED_MODEL_NAME}' — view versions/stages at {MLFLOW_URI}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
