import pandas as pd
import numpy as np
import joblib
import requests

from pathlib import Path


# =========================================================
# CONFIGURATION
# =========================================================

LATITUDE = 33.6844
LONGITUDE = 73.0479

TIMEZONE = "Asia/Karachi"

WEATHER_URL = "https://api.open-meteo.com/v1/forecast"

MODEL_PATH = Path("model/forecast_random_forest.pkl")
HISTORY_PATH = Path("data/processed/forecast_features.csv")
OUTPUT_PATH = Path("data/predictions/aqi_72h_forecast.csv")


FEATURES = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "surface_pressure",
    "wind_speed_10m",
    "wind_direction_10m",
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
    "aqi_rolling_std_24",
]


# =========================================================
# LOAD MODEL
# =========================================================

print("\nLoading Random Forest model...")

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )

model = joblib.load(MODEL_PATH)

print("Model loaded successfully!")


# =========================================================
# LOAD HISTORICAL AQI
# =========================================================

print("\nLoading historical AQI data...")

if not HISTORY_PATH.exists():
    raise FileNotFoundError(
        f"Historical dataset not found: {HISTORY_PATH}"
    )

history = pd.read_csv(HISTORY_PATH)

history["time"] = pd.to_datetime(
    history["time"],
    errors="coerce"
)

history = history.dropna(
    subset=["time", "us_aqi"]
)

history = history.sort_values(
    "time"
).reset_index(drop=True)

print(f"Historical rows: {len(history)}")

print(
    f"Historical period: "
    f"{history['time'].min()} → "
    f"{history['time'].max()}"
)


# =========================================================
# CURRENT TIME
# =========================================================

current_time = (
    pd.Timestamp.now(tz=TIMEZONE)
    .floor("h")
    .tz_localize(None)
)

print(f"\nForecast starting time: {current_time}")


# =========================================================
# GET WEATHER FORECAST
# =========================================================

print("\nFetching 72-hour weather forecast...")

weather_params = {
    "latitude": LATITUDE,
    "longitude": LONGITUDE,

    "hourly": [
        "temperature_2m",
        "relative_humidity_2m",
        "precipitation",
        "surface_pressure",
        "wind_speed_10m",
        "wind_direction_10m",
    ],

    "forecast_days": 3,

    "timezone": TIMEZONE,

    # Force hourly data
    "temperature_unit": "celsius",
    "wind_speed_unit": "kmh",
    "precipitation_unit": "mm",
}


response = requests.get(
    WEATHER_URL,
    params=weather_params,
    timeout=30
)

response.raise_for_status()

weather_data = response.json()

if "hourly" not in weather_data:
    raise ValueError(
        "Open-Meteo response does not contain hourly data."
    )

weather_df = pd.DataFrame(
    weather_data["hourly"]
)

weather_df["time"] = pd.to_datetime(
    weather_df["time"],
    errors="coerce"
)

weather_df = weather_df.dropna(
    subset=["time"]
)

weather_df = weather_df.sort_values(
    "time"
).reset_index(drop=True)


print(
    f"Weather timestamps received: "
    f"{len(weather_df)}"
)

print(
    f"Weather period: "
    f"{weather_df['time'].min()} → "
    f"{weather_df['time'].max()}"
)


# =========================================================
# SELECT FUTURE WEATHER
# =========================================================

future_weather = weather_df[
    weather_df["time"] >= current_time
].copy()

future_weather = future_weather.head(72)

print(
    f"\nFuture weather timestamps available: "
    f"{len(future_weather)}"
)


# =========================================================
# FALLBACK FOR TIME DIFFERENCE
# =========================================================

# Sometimes the local machine time and Open-Meteo
# timestamps can differ by one or more hours.
#
# If fewer than 72 future timestamps are found,
# start from the first weather timestamp after
# the latest historical observation.

if len(future_weather) < 72:

    print(
        "\nCurrent-time alignment did not provide "
        "72 timestamps."
    )

    latest_history_time = history["time"].max()

    fallback_weather = weather_df[
        weather_df["time"] > latest_history_time
    ].copy()

    fallback_weather = fallback_weather.head(72)

    if len(fallback_weather) >= len(future_weather):

        future_weather = fallback_weather

        print(
            "Using historical-data alignment "
            "for forecast timestamps."
        )


# =========================================================
# FINAL VALIDATION
# =========================================================

if len(future_weather) == 0:

    raise ValueError(
        "\nNo future weather data is available.\n"
        "Check the timestamps returned by Open-Meteo."
    )


if len(future_weather) < 72:

    print(
        f"\nWARNING: Only "
        f"{len(future_weather)} future weather "
        f"timestamps are available."
    )

    print(
        "The script will forecast the available "
        "hours instead of producing invalid data."
    )


# =========================================================
# PREPARE AQI HISTORY
# =========================================================

historical_aqi = history[
    history["time"] < future_weather["time"].min()
].copy()

historical_aqi = historical_aqi.sort_values(
    "time"
)

aqi_history = list(
    historical_aqi["us_aqi"].astype(float)
)


print(
    f"\nHistorical AQI observations available: "
    f"{len(aqi_history)}"
)


if len(aqi_history) < 24:

    raise ValueError(
        "At least 24 historical AQI observations "
        "are required for the 24-hour lag features."
    )


# =========================================================
# FEATURE CREATION
# =========================================================

def create_features(
    timestamp,
    weather_row,
    aqi_values
):

    features = {

        # -----------------------------
        # Weather
        # -----------------------------

        "temperature_2m":
            weather_row["temperature_2m"],

        "relative_humidity_2m":
            weather_row["relative_humidity_2m"],

        "precipitation":
            weather_row["precipitation"],

        "surface_pressure":
            weather_row["surface_pressure"],

        "wind_speed_10m":
            weather_row["wind_speed_10m"],

        "wind_direction_10m":
            weather_row["wind_direction_10m"],


        # -----------------------------
        # Time
        # -----------------------------

        "hour":
            timestamp.hour,

        "day_of_week":
            timestamp.dayofweek,

        "month":
            timestamp.month,

        "is_weekend":
            int(timestamp.dayofweek >= 5),


        # -----------------------------
        # AQI Lags
        # -----------------------------

        "aqi_lag_1":
            aqi_values[-1],

        "aqi_lag_3":
            aqi_values[-3],

        "aqi_lag_6":
            aqi_values[-6],

        "aqi_lag_12":
            aqi_values[-12],

        "aqi_lag_24":
            aqi_values[-24],


        # -----------------------------
        # Rolling statistics
        # -----------------------------

        "aqi_rolling_mean_6":
            np.mean(aqi_values[-6:]),

        "aqi_rolling_mean_24":
            np.mean(aqi_values[-24:]),

        "aqi_rolling_std_24":
            np.std(aqi_values[-24:]),
    }

    return pd.DataFrame(
        [features],
        columns=FEATURES
    )


# =========================================================
# GENERATE FORECAST
# =========================================================

print("\nGenerating AQI forecasts...")

predictions = []


for _, weather_row in future_weather.iterrows():

    timestamp = weather_row["time"]

    X = create_features(
        timestamp,
        weather_row,
        aqi_history
    )

    # Make prediction
    prediction = model.predict(X)[0]

    # AQI cannot be negative
    prediction = max(
        0,
        float(prediction)
    )

    # Save prediction
    predictions.append({

        "time":
            timestamp,

        "predicted_aqi":
            round(prediction, 2)
    })

    # Important:
    # Feed prediction back into history
    # so the next hour can use it as lag_1.
    aqi_history.append(prediction)


# =========================================================
# CREATE FORECAST DATAFRAME
# =========================================================

forecast_df = pd.DataFrame(
    predictions
)


# =========================================================
# AQI CATEGORY
# =========================================================

def get_aqi_category(aqi):

    if aqi <= 50:
        return "Good"

    elif aqi <= 100:
        return "Moderate"

    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups"

    elif aqi <= 200:
        return "Unhealthy"

    elif aqi <= 300:
        return "Very Unhealthy"

    else:
        return "Hazardous"


forecast_df["category"] = (
    forecast_df["predicted_aqi"]
    .apply(get_aqi_category)
)


# =========================================================
# SAVE FORECAST
# =========================================================

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

forecast_df.to_csv(
    OUTPUT_PATH,
    index=False
)


# =========================================================
# DISPLAY RESULTS
# =========================================================

print("\n========== 72-HOUR AQI FORECAST ==========\n")

print(
    forecast_df.to_string(index=False)
)

print(
    f"\nTotal forecast hours: "
    f"{len(forecast_df)}"
)

print(
    f"\nMinimum predicted AQI: "
    f"{forecast_df['predicted_aqi'].min():.2f}"
)

print(
    f"Maximum predicted AQI: "
    f"{forecast_df['predicted_aqi'].max():.2f}"
)

print(
    f"Average predicted AQI: "
    f"{forecast_df['predicted_aqi'].mean():.2f}"
)

print(
    f"\nForecast saved to:"
    f"\n{OUTPUT_PATH}"
)

print(
    "\n72-hour forecasting completed successfully!"
)