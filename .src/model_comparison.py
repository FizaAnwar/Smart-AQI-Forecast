import pandas as pd
from pathlib import Path

OUTPUT_PATH = Path("data/predictions/model_comparison.csv")

results = {
    "Model": [
        "Ridge Regression",
        "Random Forest",
        "LSTM"
    ],
    "MAE": [
        5.308,
        4.909,
        17.869
    ],
    "RMSE": [
        7.821,
        7.911,
        24.388
    ],
    "R2": [
        0.874,
        0.871,
        -0.236
    ]
}

df = pd.DataFrame(results)

# Rank models by MAE
df["MAE_Rank"] = df["MAE"].rank(method="min")

df = df.sort_values("MAE").reset_index(drop=True)

print("\n========== MODEL COMPARISON ==========")
print(df.to_string(index=False))

# Save results
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

df.to_csv(OUTPUT_PATH, index=False)

print(f"\nComparison saved to: {OUTPUT_PATH}")

print("\nBest model based on MAE:")
print(df.iloc[0]["Model"])