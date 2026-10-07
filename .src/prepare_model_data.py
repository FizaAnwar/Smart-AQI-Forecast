import pandas as pd
from pathlib import Path

INPUT_PATH = Path("data/processed/forecast_features.csv")

TRAIN_PATH = Path("data/processed/forecast_train.csv")
TEST_PATH = Path("data/processed/forecast_test.csv")


def prepare_data():

    df = pd.read_csv(INPUT_PATH)
    df["time"] = pd.to_datetime(df["time"])

    # Always sort chronologically
    df = df.sort_values("time").reset_index(drop=True)

    # 80% past data → training
    # 20% future data → testing
    split_index = int(len(df) * 0.8)

    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()

    return train_df, test_df


def save_data(train_df, test_df):

    TRAIN_PATH.parent.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(TRAIN_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)

    print(f"Training data saved: {TRAIN_PATH}")
    print(f"Testing data saved: {TEST_PATH}")


if __name__ == "__main__":

    print("Preparing forecasting datasets...")

    train_df, test_df = prepare_data()

    print("\n========== DATA SPLIT ==========")

    print(f"Training rows: {len(train_df)}")
    print(f"Testing rows: {len(test_df)}")

    print("\nTraining period:")
    print(train_df["time"].min())
    print("to")
    print(train_df["time"].max())

    print("\nTesting period:")
    print(test_df["time"].min())
    print("to")
    print(test_df["time"].max())

    save_data(train_df, test_df)

    print("\nForecasting data preparation completed!")