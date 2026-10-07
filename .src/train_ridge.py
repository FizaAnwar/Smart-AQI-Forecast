import pandas as pd
import joblib

from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Paths
TRAIN_PATH = Path("data/processed/train.csv")
TEST_PATH = Path("data/processed/test.csv")
MODEL_PATH = Path("model/ridge_model.pkl")


# Load data
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

# Columns that should NOT be used as model features
DROP_COLUMNS = ["time", "us_aqi"]

X_train = train_df.drop(columns=DROP_COLUMNS)
y_train = train_df["us_aqi"]

X_test = test_df.drop(columns=DROP_COLUMNS)
y_test = test_df["us_aqi"]


# Ridge Regression pipeline
model = Pipeline([
    ("scaler", StandardScaler()),
    ("ridge", Ridge(alpha=1.0))
])


# Train
print("Training Ridge Regression model...")

model.fit(X_train, y_train)


# Predict
predictions = model.predict(X_test)


# Evaluation
mae = mean_absolute_error(y_test, predictions)
rmse = mean_squared_error(y_test, predictions) ** 0.5
r2 = r2_score(y_test, predictions)


print("\n========== RIDGE REGRESSION RESULTS ==========")
print(f"MAE  : {mae:.3f}")
print(f"RMSE : {rmse:.3f}")
print(f"R²   : {r2:.3f}")


# Save model
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

joblib.dump(model, MODEL_PATH)

print(f"\nModel saved to: {MODEL_PATH}")