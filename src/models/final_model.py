from pathlib import Path
import joblib

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.features.preprocessing import build_preprocessor



RANDOM_STATE = 42


def build_pipeline(classifier, X):
    preprocessor = build_preprocessor(X)

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def main():
    train_path = Path("data/processed/train.csv")
    test_path = Path("data/processed/test.csv")
    tuning_path = Path("data/processed/hyperparameter_results.csv")

    for path in [train_path, test_path, tuning_path]:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df.drop(columns=["Churn"])
    y_train = train_df["Churn"].map({"No": 0, "Yes": 1})

    X_test = test_df.drop(columns=["Churn"])
    y_test = test_df["Churn"].map({"No": 0, "Yes": 1})

    tuning_results = pd.read_csv(tuning_path)

    print("\n===== TUNING RESULTS =====")
    print(tuning_results[["Model", "Best ROC-AUC"]].to_string(index=False))

    # Select model based on best ROC-AUC
    best_model_name = tuning_results.loc[
        tuning_results["Best ROC-AUC"].idxmax(), "Model"
    ]

    print(f"\nSelected model: {best_model_name}")

    # Reconstruct best model using saved hyperparameters
    if best_model_name == "Random Forest":
        best_parameters = eval(
            tuning_results.loc[
                tuning_results["Model"] == "Random Forest",
                "Best Parameters",
            ].iloc[0]
        )

        classifier = RandomForestClassifier(
            n_estimators=best_parameters["classifier__n_estimators"],
            max_depth=best_parameters["classifier__max_depth"],
            min_samples_split=best_parameters[
                "classifier__min_samples_split"
            ],
            min_samples_leaf=best_parameters[
                "classifier__min_samples_leaf"
            ],
            max_features=best_parameters["classifier__max_features"],
            class_weight=best_parameters[
                "classifier__class_weight"
            ],
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    else:
        best_parameters = eval(
            tuning_results.loc[
                tuning_results["Model"] == "XGBoost",
                "Best Parameters",
            ].iloc[0]
        )

        classifier = XGBClassifier(
            n_estimators=best_parameters["classifier__n_estimators"],
            max_depth=best_parameters["classifier__max_depth"],
            learning_rate=best_parameters[
                "classifier__learning_rate"
            ],
            subsample=best_parameters["classifier__subsample"],
            colsample_bytree=best_parameters[
                "classifier__colsample_bytree"
            ],
            min_child_weight=best_parameters[
                "classifier__min_child_weight"
            ],
            random_state=RANDOM_STATE,
            eval_metric="logloss",
            n_jobs=-1,
        )

    # Build final pipeline
    final_model = build_pipeline(classifier, X_train)

    print("\n===== TRAINING FINAL MODEL =====")

    final_model.fit(X_train, y_train)

    model_dir = Path("models")
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "final_model.joblib"

    joblib.dump(final_model, model_path)

    print(f"Final model saved to: {model_path}")

    print("Final model training completed.")

    # Test evaluation
    y_pred = final_model.predict(X_test)
    y_prob = final_model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    print("\n===== FINAL TEST RESULTS =====")

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")
    print(f"PR-AUC   : {pr_auc:.4f}")

    print("\n===== CONFUSION MATRIX =====")
    print(cm)

    # Save metrics
    metrics = pd.DataFrame(
        [
            {
                "Model": best_model_name,
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1": f1,
                "ROC-AUC": roc_auc,
                "PR-AUC": pr_auc,
            }
        ]
    )

    output_path = Path("data/processed/final_test_results.csv")
    metrics.to_csv(output_path, index=False)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()