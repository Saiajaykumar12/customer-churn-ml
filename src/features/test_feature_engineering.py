from preprocessing import clean_data, load_data
from feature_engineering import create_features


def main() -> None:
    df = load_data()

    df = clean_data(df)

    df = create_features(df)

    new_features = [
        "AverageMonthlySpend",
        "ServiceCount",
        "TenureGroup",
        "IsMonthToMonth",
    ]

    print("\nCreated features:")

    for feature in new_features:
        print(f"\n{feature}:")
        print(df[feature].head())

    print("\nFinal shape:", df.shape)


if __name__ == "__main__":
    main()