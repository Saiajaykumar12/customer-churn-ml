from sklearn.model_selection import train_test_split

from preprocessing import build_preprocessor, clean_data, load_data


def main() -> None:
    df = load_data()

    print("Original shape:", df.shape)

    df = clean_data(df)

    print("Cleaned shape:", df.shape)

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    preprocessor = build_preprocessor(X_train)

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    print("\nTraining data shape:", X_train.shape)
    print("Test data shape:", X_test.shape)

    print(
        "Processed training shape:",
        X_train_processed.shape,
    )

    print(
        "Processed test shape:",
        X_test_processed.shape,
    )


if __name__ == "__main__":
    main()