from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.features.preprocessing import build_preprocessor


RANDOM_STATE = 42
N_SPLITS = 5


def build_model(classifier, X):
    preprocessor = build_preprocessor(X)

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def main():
    train_path = Path("data/processed/train.csv")

    if not train_path.exists():
        raise FileNotFoundError(f"Training data not found: {train_path}")

    df = pd.read_csv(train_path)

    X = df.drop(columns=["Churn"])
    y = df["Churn"].map({"No": 0, "Yes": 1})

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            eval_metric="logloss",
        ),
    }

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision",
    }

    results = []

    for name, classifier in models.items():
        print(f"\n===== {name} =====")

        pipeline = build_model(classifier, X)

        cv_results = cross_validate(
            pipeline,
            X,
            y,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
        )

        result = {
            "Model": name,
        }

        for metric in scoring:
            scores = cv_results[f"test_{metric}"]

            result[f"{metric}_mean"] = scores.mean()
            result[f"{metric}_std"] = scores.std()

            print(
                f"{metric.upper():<10}: "
                f"{scores.mean():.4f} +/- {scores.std():.4f}"
            )

        results.append(result)

    results_df = pd.DataFrame(results)

    print("\n===== CROSS-VALIDATION SUMMARY =====")
    print(results_df.to_string(index=False))

    output_path = Path("data/processed/cross_validation_results.csv")
    results_df.to_csv(output_path, index=False)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()