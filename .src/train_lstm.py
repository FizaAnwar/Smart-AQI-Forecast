import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


# --------------------------------------------------
# Paths
# --------------------------------------------------

TRAIN_PATH = Path("data/processed/forecast_train.csv")
TEST_PATH = Path("data/processed/forecast_test.csv")

MODEL_PATH = Path("model/forecast_lstm.keras")
SCALER_PATH = Path("model/lstm_scaler.pkl")


# --------------------------------------------------
# Load data
# --------------------------------------------------

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

train_df["time"] = pd.to_datetime(train_df["time"])
test_df["time"] = pd.to_datetime(test_df["time"])

train_df = train_df.sort_values("time").reset_index(drop=True)
test_df = test_df.sort_values("time").reset_index(drop=True)


# --------------------------------------------------
# Select features
# --------------------------------------------------

TARGET = "us_aqi"

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
    "aqi_rolling_std_24"
]


X_train = train_df[FEATURES].values
y_train = train_df[TARGET].values

X_test = test_df[FEATURES].values
y_test = test_df[TARGET].values


print("Number of features:", len(FEATURES))
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# --------------------------------------------------
# Scale features
# --------------------------------------------------

scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# --------------------------------------------------
# Create sequences
# --------------------------------------------------

SEQUENCE_LENGTH = 24


def create_sequences(X, y, sequence_length):

    X_sequences = []
    y_sequences = []

    for i in range(sequence_length, len(X)):

        X_sequences.append(
            X[i - sequence_length:i]
        )

        y_sequences.append(y[i])

    return np.array(X_sequences), np.array(y_sequences)


X_train_seq, y_train_seq = create_sequences(
    X_train_scaled,
    y_train,
    SEQUENCE_LENGTH
)

X_test_seq, y_test_seq = create_sequences(
    X_test_scaled,
    y_test,
    SEQUENCE_LENGTH
)


print("\nSequence shape:")
print(X_train_seq.shape)


# --------------------------------------------------
# Build LSTM model
# --------------------------------------------------

model = Sequential([

    LSTM(
        64,
        return_sequences=True,
        input_shape=(SEQUENCE_LENGTH, len(FEATURES))
    ),

    Dropout(0.2),

    LSTM(32),

    Dropout(0.2),

    Dense(16, activation="relu"),

    Dense(1)
])


model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)


model.summary()


# --------------------------------------------------
# Early stopping
# --------------------------------------------------

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)


# --------------------------------------------------
# Train
# --------------------------------------------------

print("\nTraining LSTM...")

history = model.fit(
    X_train_seq,
    y_train_seq,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=1
)


# --------------------------------------------------
# Predictions
# --------------------------------------------------

predictions = model.predict(
    X_test_seq,
    verbose=0
).flatten()


# --------------------------------------------------
# Evaluation
# --------------------------------------------------

mae = mean_absolute_error(
    y_test_seq,
    predictions
)

rmse = mean_squared_error(
    y_test_seq,
    predictions
) ** 0.5

r2 = r2_score(
    y_test_seq,
    predictions
)


print("\n========== LSTM RESULTS ==========")

print(f"MAE  : {mae:.3f}")
print(f"RMSE : {rmse:.3f}")
print(f"R²   : {r2:.3f}")


# --------------------------------------------------
# Save model and scaler
# --------------------------------------------------

MODEL_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

model.save(MODEL_PATH)

joblib.dump(
    scaler,
    SCALER_PATH
)


print(f"\nLSTM model saved to: {MODEL_PATH}")
print(f"Scaler saved to: {SCALER_PATH}")

print("\nLSTM training completed!")