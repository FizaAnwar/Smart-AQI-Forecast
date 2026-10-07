import pandas as pd
from pathlib import Path

INPUT_PATH = Path("data/processed/islamabad_aqi_clean.csv")
OUTPUT_PATH = Path("data/processed/forecast_features.csv")


def create_forecasting_features(df):

    df = df.sort_values("time").copy()

    # Time features
    df["hour"] = df["time"].dt.hour
    df["day_of_week"] = df["time"].dt.dayofweek
    df["month"] = df["time"].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # Historical AQI features
    df["aqi_lag_1"] = df["us_aqi"].shift(1)
    df["aqi_lag_3"] = df["us_aqi"].shift(3)
    df["aqi_lag_6"] = df["us_aqi"].shift(6)
    df["aqi_lag_12"] = df["us_aqi"].shift(12)
    df["aqi_lag_24"] = df["us_aqi"].shift(24)

    # Historical rolling statistics
    df["aqi_rolling_mean_6"] = (
        df["us_aqi"].shift(1).rolling(6).mean()
    )

    df["aqi_rolling_mean_24"] = (
        df["us_aqi"].shift(1).rolling(24).mean()
    )

    df["aqi_rolling_std_24"] = (
        df["us_aqi"].shift(1).rolling(24).std()
    )

    # Weather variables are retained because they can be
    # obtained from weather forecasts for future timestamps.
    weather_features = [
        "temperature_2m",
        "relative_humidity_2m",
        "precipitation",
        "surface_pressure",
        "wind_speed_10m",
        "wind_direction_10m"
    ]

    # Keep only required forecasting columns
    feature_columns = [
        "time",
        "us_aqi"
    ] + weather_features + [
        "hour",
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
        "aqi_rolling_std_24"
    ]

    df = df[feature_columns]

    # Remove rows where lag/rolling features aren't available
    df = df.dropna().reset_index(drop=True)

    return df


if __name__ == "__main__":

    print("Loading cleaned AQI dataset...")

    df = pd.read_csv(INPUT_PATH)
    df["time"] = pd.to_datetime(df["time"])

    print(f"Original rows: {len(df)}")

    forecast_df = create_forecasting_features(df)

    print(f"Rows after feature engineering: {len(forecast_df)}")
    print(f"Columns: {len(forecast_df.columns)}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    forecast_df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nForecasting dataset saved to:")
    print(OUTPUT_PATH)

    print("\nFeature columns:")
    print(forecast_df.columns.tolist())

    print("\nForecasting feature preparation completed!")