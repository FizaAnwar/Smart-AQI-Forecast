import pandas as pd
from pathlib import Path


RAW_PATH = Path("data/raw/islamabad_aqi_weather.csv")
PROCESSED_PATH = Path("data/processed/islamabad_aqi_clean.csv")


def load_data():
    """Load the raw AQI dataset."""
    df = pd.read_csv(RAW_PATH)

    df["time"] = pd.to_datetime(df["time"])

    return df


def inspect_data(df):
    """Display basic dataset information."""

    print("\n========== DATASET INFO ==========")

    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nDuplicate rows:")
    print(df.duplicated().sum())

    print("\nData types:")
    print(df.dtypes)


def clean_data(df):
    """Clean and prepare the dataset."""

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Sort chronologically
    df = df.sort_values("time")

    # Remove rows where AQI is missing
    df = df.dropna(subset=["us_aqi"])

    # Interpolate missing numeric values
    numeric_columns = df.select_dtypes(
        include=["number"]
    ).columns

    df[numeric_columns] = (
        df[numeric_columns]
        .interpolate(method="linear")
    )

    # Remove remaining missing rows
    df = df.dropna()

    return df


def save_data(df):
    """Save cleaned dataset."""

    PROCESSED_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        PROCESSED_PATH,
        index=False
    )

    print(
        f"\nClean dataset saved to: "
        f"{PROCESSED_PATH}"
    )


if __name__ == "__main__":

    print("Loading raw dataset...")

    df = load_data()

    inspect_data(df)

    print("\nCleaning dataset...")

    cleaned_df = clean_data(df)

    print("\n========== CLEAN DATA ==========")

    print(f"Rows after cleaning: {len(cleaned_df)}")
    print(f"Columns: {len(cleaned_df.columns)}")

    print("\nRemaining missing values:")
    print(cleaned_df.isnull().sum())

    save_data(cleaned_df)

    print("\nData preprocessing completed!")