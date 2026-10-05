from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.features.feature_engineering import create_features
from src.features.preprocessing import clean_data, load_data


RANDOM_STATE = 42
TEST_SIZE = 0.20


def main() -> None:
    df = load_data()

    df = clean_data(df)
    df = create_features(df)

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    train_data = X_train.copy()
    train_data["Churn"] = y_train

    test_data = X_test.copy()
    test_data["Churn"] = y_test

    train_data.to_csv(output_dir / "train.csv", index=False)
    test_data.to_csv(output_dir / "test.csv", index=False)

    print("Train shape:", X_train.shape)
    print("Test shape:", X_test.shape)

    print("\nTrain target distribution:")
    print(y_train.value_counts(normalize=True))

    print("\nTest target distribution:")
    print(y_test.value_counts(normalize=True))

    print("\nSaved:")
    print("data/processed/train.csv")
    print("data/processed/test.csv")


if __name__ == "__main__":
    main()