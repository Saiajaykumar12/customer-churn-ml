from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd


MODEL_PATH = Path("models/final_model.joblib")
RESULTS_PATH = Path("data/processed/final_test_results.csv")
MLFLOW_DB = Path("mlflow.db")


def main():
    # Check required files
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    if not RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"Results not found: {RESULTS_PATH}"
        )

    # Load final model and evaluation results
    model = joblib.load(MODEL_PATH)
    results = pd.read_csv(RESULTS_PATH)

    metrics = results.iloc[0]

    # SQLite tracking backend
    tracking_uri = f"sqlite:///{MLFLOW_DB.resolve().as_posix()}"

    mlflow.set_tracking_uri(tracking_uri)

    # Create or use experiment
    mlflow.set_experiment("customer-churn-prediction")

    with mlflow.start_run(run_name="final-churn-model"):

        classifier = model.named_steps["classifier"]

        # -----------------------------
        # Model parameters
        # -----------------------------

        mlflow.log_param(
            "model_type",
            type(classifier).__name__,
        )

        mlflow.log_param(
            "random_state",
            42,
        )

        # -----------------------------
        # Final test metrics
        # -----------------------------

        mlflow.log_metric(
            "accuracy",
            float(metrics["Accuracy"]),
        )

        mlflow.log_metric(
            "precision",
            float(metrics["Precision"]),
        )

        mlflow.log_metric(
            "recall",
            float(metrics["Recall"]),
        )

        mlflow.log_metric(
            "f1",
            float(metrics["F1"]),
        )

        mlflow.log_metric(
            "roc_auc",
            float(metrics["ROC-AUC"]),
        )

        mlflow.log_metric(
            "pr_auc",
            float(metrics["PR-AUC"]),
        )

        # -----------------------------
        # Log complete ML pipeline
        # -----------------------------

        trusted_types = [
            "numpy.dtype",
            "xgboost.core.Booster",
            "xgboost.sklearn.XGBClassifier",
        ]

        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            skops_trusted_types=trusted_types,
        )

        # -----------------------------
        # Log evaluation results
        # -----------------------------

        mlflow.log_artifact(
            str(RESULTS_PATH),
            artifact_path="evaluation",
        )

        # -----------------------------
        # Log SHAP plots
        # -----------------------------

        shap_dir = Path("models/shap")

        if shap_dir.exists():
            for file in shap_dir.glob("*.png"):
                mlflow.log_artifact(
                    str(file),
                    artifact_path="shap",
                )

        run_id = mlflow.active_run().info.run_id

    print("\n===== MLFLOW RUN COMPLETE =====")
    print("Experiment: customer-churn-prediction")
    print(f"Run ID: {run_id}")
    print(f"Tracking database: {MLFLOW_DB}")
    print("Model artifact: logged")
    print("Evaluation metrics: logged")
    print("SHAP artifacts: logged")


if __name__ == "__main__":
    main()