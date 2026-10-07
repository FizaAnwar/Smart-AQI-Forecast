import pandas as pd
from pathlib import Path


INPUT_PATH = Path("data/processed/islamabad_aqi_clean.csv")
OUTPUT_PATH = Path("data/processed/islamabad_aqi_features.csv")


def load_data():
    """Load cleaned AQI data."""

    df = pd.read_csv(INPUT_PATH)

    df["time"] = pd.to_datetime(df["time"])

    return df


def create_time_features(df):
    """Create calendar and time-based features."""

    df["hour"] = df["time"].dt.hour
    df["day"] = df["time"].dt.day
    df["day_of_week"] = df["time"].dt.dayofweek
    df["month"] = df["time"].dt.month

    # Monday-Friday = 0, Saturday/Sunday = 1
    df["is_weekend"] = (
        df["day_of_week"] >= 5
    ).astype(int)

    return df


def create_lag_features(df):
    """Create previous AQI values."""

    df["aqi_lag_1"] = df["us_aqi"].shift(1)
    df["aqi_lag_3"] = df["us_aqi"].shift(3)
    df["aqi_lag_6"] = df["us_aqi"].shift(6)
    df["aqi_lag_12"] = df["us_aqi"].shift(12)
    df["aqi_lag_24"] = df["us_aqi"].shift(24)

    return df


def create_rolling_features(df):
    """Create rolling AQI statistics."""

    df["aqi_rolling_mean_6"] = (
        df["us_aqi"]
        .rolling(window=6)
        .mean()
    )

    df["aqi_rolling_mean_24"] = (
        df["us_aqi"]
        .rolling(window=24)
        .mean()
    )

    df["aqi_rolling_std_24"] = (
        df["us_aqi"]
        .rolling(window=24)
        .std()
    )

    return df


def create_change_features(df):
    """Measure AQI changes over time."""

    df["aqi_change"] = (
        df["us_aqi"].diff()
    )

    df["aqi_change_rate"] = (
        df["us_aqi"]
        .pct_change()
        .replace([float("inf"), -float("inf")], 0)
    )

    return df


def create_features(df):
    """Run all feature engineering steps."""

    df = df.sort_values("time").copy()

    df = create_time_features(df)
    df = create_lag_features(df)
    df = create_rolling_features(df)
    df = create_change_features(df)

    return df


def save_features(df):
    """Save feature-engineered dataset."""

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print(
        f"\nFeature dataset saved to: "
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":

    print("Loading cleaned dataset...")

    df = load_data()

    print(f"Original rows: {len(df)}")

    df = create_features(df)

    # Lag and rolling features create NaN values
    df = df.dropna().reset_index(drop=True)

    print(
        f"Rows after feature engineering: "
        f"{len(df)}"
    )

    print(
        f"Total features: "
        f"{len(df.columns)}"
    )

    print("\nNew feature columns:")

    feature_columns = [
        "hour",
        "day",
        "day_of_week",
        "month",
        "is_weekend",
        "aqi_lag_1",
        "aqi_lag_3",
        "aqi_lag_6",
        "aqi_lag_12",
        "aqi_lag_24",
        "aqi_rolling_mean_6",
        "aqi_rolling_mean_24",
        "aqi_rolling_std_24",
        "aqi_change",
        "aqi_change_rate"
    ]

    print(feature_columns)

    save_features(df)

    print("\nFeature engineering completed!")