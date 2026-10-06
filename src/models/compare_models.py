from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.features.preprocessing import build_preprocessor


RANDOM_STATE = 42
VALIDATION_SIZE = 0.20


def evaluate_model(name, model, X_train, X_val, y_train, y_val):
    print(f"\n===== {name} =====")

    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]

    results = {
        "Model": name,
        "Accuracy": accuracy_score(y_val, y_pred),
        "Precision": precision_score(y_val, y_pred),
        "Recall": recall_score(y_val, y_pred),
        "F1": f1_score(y_val, y_pred),
        "ROC-AUC": roc_auc_score(y_val, y_prob),
        "PR-AUC": average_precision_score(y_val, y_prob),
    }

    for metric, value in results.items():
        if metric != "Model":
            print(f"{metric:<10}: {value:.4f}")

    return results


def main():
    train_path = Path("data/processed/train.csv")

    if not train_path.exists():
        raise FileNotFoundError(f"Training data not found: {train_path}")

    df = pd.read_csv(train_path)

    X = df.drop(columns=["Churn"])
    y = df["Churn"].map({"No": 0, "Yes": 1})

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

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

    results = []

    for name, classifier in models.items():
        preprocessor = build_preprocessor(X_train)

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", classifier),
            ]
        )

        result = evaluate_model(
            name,
            pipeline,
            X_train,
            X_val,
            y_train,
            y_val,
        )

        results.append(result)

    results_df = pd.DataFrame(results)

    print("\n===== MODEL COMPARISON =====")
    print(results_df.to_string(index=False))

    output_path = Path("data/processed/model_comparison.csv")
    results_df.to_csv(output_path, index=False)

    print(f"\nComparison saved to: {output_path}")


if __name__ == "__main__":
    main()