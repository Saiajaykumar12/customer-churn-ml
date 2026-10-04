from pathlib import Path

import pandas as pd


DATA_PATH = Path("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    print("\n===== DATASET SHAPE =====")
    print(df.shape)

    print("\n===== COLUMNS =====")
    print(df.columns.tolist())

    print("\n===== DATA TYPES =====")
    print(df.dtypes)

    print("\n===== MISSING VALUES =====")
    print(df.isnull().sum())

    print("\n===== DUPLICATES =====")
    print(df.duplicated().sum())

    print("\n===== UNIQUE VALUES =====")
    for column in df.columns:
        print(f"\n{column}:")
        print(df[column].unique()[:10])

    print("\n===== BASIC STATISTICS =====")
    print(df.describe(include="all").transpose())

    print("\n===== TARGET DISTRIBUTION =====")
    print(df["Churn"].value_counts())

    print("\n===== TARGET PERCENTAGE =====")
    print(df["Churn"].value_counts(normalize=True) * 100)


if __name__ == "__main__":
    main()