import pandas as pd
import joblib

from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


TRAIN_PATH = Path("data/processed/forecast_train.csv")
TEST_PATH = Path("data/processed/forecast_test.csv")
MODEL_PATH = Path("model/forecast_ridge_model.pkl")


# Load datasets
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

train_df["time"] = pd.to_datetime(train_df["time"])
test_df["time"] = pd.to_datetime(test_df["time"])


# Target
TARGET = "us_aqi"

# Remove timestamp and target from features
DROP_COLUMNS = ["time", TARGET]

X_train = train_df.drop(columns=DROP_COLUMNS)
y_train = train_df[TARGET]

X_test = test_df.drop(columns=DROP_COLUMNS)
y_test = test_df[TARGET]


print("Training features:")
print(X_train.columns.tolist())

print("\nNumber of features:", X_train.shape[1])


# Ridge Regression
model = Pipeline([
    ("scaler", StandardScaler()),
    ("ridge", Ridge(alpha=1.0))
])


# Train
print("\nTraining leakage-free Ridge Regression...")

model.fit(X_train, y_train)


# Predict
predictions = model.predict(X_test)


# Evaluation
mae = mean_absolute_error(y_test, predictions)
rmse = mean_squared_error(y_test, predictions) ** 0.5
r2 = r2_score(y_test, predictions)


print("\n========== FORECAST RIDGE RESULTS ==========")
print(f"MAE  : {mae:.3f}")
print(f"RMSE : {rmse:.3f}")
print(f"R²   : {r2:.3f}")


# Save model
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

joblib.dump(model, MODEL_PATH)

print(f"\nModel saved to: {MODEL_PATH}")