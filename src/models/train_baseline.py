from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.features.preprocessing import build_preprocessor


RANDOM_STATE = 42
VALIDATION_SIZE = 0.20


def main():
    # Load training data
    train_path = Path("data/processed/train.csv")

    if not train_path.exists():
        raise FileNotFoundError(f"Training data not found: {train_path}")

    df = pd.read_csv(train_path)

    # Separate features and target
    X = df.drop(columns=["Churn"])
    y = df["Churn"].map({"No": 0, "Yes": 1})

    # Train-validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # Build preprocessing + Logistic Regression pipeline
    preprocessor = build_preprocessor(X_train)

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    # Train
    print("\n===== TRAINING LOGISTIC REGRESSION =====")

    model.fit(X_train, y_train)

    print("Training completed successfully.")

    # Predictions
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]

    # Metrics
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    roc_auc = roc_auc_score(y_val, y_prob)
    pr_auc = average_precision_score(y_val, y_prob)
    cm = confusion_matrix(y_val, y_pred)

    print("\n===== VALIDATION RESULTS =====")

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")
    print(f"PR-AUC   : {pr_auc:.4f}")

    print("\n===== CONFUSION MATRIX =====")
    print(cm)

    print("\n===== DATA SHAPES =====")
    print("Training:", X_train.shape)
    print("Validation:", X_val.shape)


if __name__ == "__main__":
    main()