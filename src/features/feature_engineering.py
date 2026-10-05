import pandas as pd


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Average monthly spending
    df["AverageMonthlySpend"] = (
        df["TotalCharges"] / df["tenure"].replace(0, 1)
    )

    # Count subscribed services
    service_columns = [
        "PhoneService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]

    df["ServiceCount"] = 0

    for column in service_columns:
        df["ServiceCount"] += (
            df[column].isin(["Yes"]).astype(int)
        )

    # Tenure groups
    df["TenureGroup"] = pd.cut(
        df["tenure"],
        bins=[-1, 12, 24, 48, float("inf")],
        labels=[
            "0-12",
            "13-24",
            "25-48",
            "49+",
        ],
    )

    # Contract indicator
    df["IsMonthToMonth"] = (
        df["Contract"] == "Month-to-month"
    ).astype(int)

    return df