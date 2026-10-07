import requests
import pandas as pd
from pathlib import Path


# Islamabad coordinates
LATITUDE = 33.6844
LONGITUDE = 73.0479

AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"


def fetch_air_quality():
    """Fetch hourly air-quality data for Islamabad."""

    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "hourly": [
            "pm10",
            "pm2_5",
            "carbon_monoxide",
            "nitrogen_dioxide",
            "sulphur_dioxide",
            "ozone",
            "us_aqi"
        ],
        "past_days": 30,
        "forecast_days": 3,
        "timezone": "Asia/Karachi"
    }

    response = requests.get(AIR_QUALITY_URL, params=params, timeout=30)
    response.raise_for_status()

    return response.json()


def fetch_weather():
    """Fetch hourly weather data for Islamabad."""

    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "hourly": [
            "temperature_2m",
            "relative_humidity_2m",
            "precipitation",
            "surface_pressure",
            "wind_speed_10m",
            "wind_direction_10m"
        ],
        "past_days": 30,
        "forecast_days": 3,
        "timezone": "Asia/Karachi"
    }

    response = requests.get(WEATHER_URL, params=params, timeout=30)
    response.raise_for_status()

    return response.json()


def create_dataset():
    """Combine air-quality and weather data."""

    air_quality = fetch_air_quality()
    weather = fetch_weather()

    air_df = pd.DataFrame(air_quality["hourly"])
    weather_df = pd.DataFrame(weather["hourly"])

    dataset = pd.merge(
        air_df,
        weather_df,
        on="time",
        how="inner"
    )

    dataset["time"] = pd.to_datetime(dataset["time"])

    return dataset


def save_dataset(dataset):
    """Save raw dataset as CSV."""

    output_path = Path("data/raw/islamabad_aqi_weather.csv")

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    dataset.to_csv(
        output_path,
        index=False
    )

    print(f"Dataset saved to: {output_path}")
    print(f"Rows: {len(dataset)}")
    print(f"Columns: {len(dataset.columns)}")


if __name__ == "__main__":

    print("Fetching Islamabad AQI and weather data...")

    dataset = create_dataset()

    print("\nDataset preview:")
    print(dataset.head())

    print("\nDataset columns:")
    print(dataset.columns.tolist())

    save_dataset(dataset)

    print("\nData collection completed successfully!")