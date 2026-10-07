import pandas as pd
from pathlib import Path

# Paths
INPUT_PATH = Path("data/processed/islamabad_aqi_features.csv")
TRAIN_PATH = Path("data/processed/train.csv")
TEST_PATH = Path("data/processed/test.csv")


# Load dataset
def load_data():
    df = pd.read_csv(INPUT_PATH)
    df["time"] = pd.to_datetime(df["time"])

    return df


# Create time-based train/test split
def split_data(df, test_size=0.2):

    df = df.sort_values("time").reset_index(drop=True)

    split_index = int(len(df) * (1 - test_size))

    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()

    return train_df, test_df


# Save datasets
def save_data(train_df, test_df):

    TRAIN_PATH.parent.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(TRAIN_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)

    print(f"Training data saved to: {TRAIN_PATH}")
    print(f"Testing data saved to: {TEST_PATH}")


if __name__ == "__main__":

    print("Loading feature dataset...")

    df = load_data()

    print(f"Total rows: {len(df)}")

    train_df, test_df = split_data(df)

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

    print("\nModel preparation completed!")