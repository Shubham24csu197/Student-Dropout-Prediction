
import pandas as pd
import json

from schemas import (
    CATEGORICAL_FEATURES,
    ORDINAL_FEATURES,
    NUMERICAL_FEATURES
)

# Load cleaned dataset
df = pd.read_csv("data/student_dropout_cleaned.csv")

contract = {}

# Inspect categorical features
for feature in CATEGORICAL_FEATURES:
    values = sorted(df[feature].dropna().unique().tolist())

    contract[feature] = {
        "type": "categorical",
        "allowed_values": [int(v) for v in values]
    }

# Inspect ordinal features
for feature in ORDINAL_FEATURES:
    contract[feature] = {
        "type": "ordinal",
        "minimum": int(df[feature].min()),
        "maximum": int(df[feature].max()),
        "allowed_values": sorted(
            [int(v) for v in df[feature].dropna().unique()]
        )
    }

# Inspect numerical features
for feature in NUMERICAL_FEATURES:
    contract[feature] = {
        "type": "numerical",
        "minimum": float(df[feature].min()),
        "maximum": float(df[feature].max())
    }

# Save report
with open("data_contract.json", "w") as file:
    json.dump(contract, file, indent=4)

print("DATA CONTRACT REPORT")
print("=" * 60)

for feature, details in contract.items():
    print(f"\n{feature}")
    print(details)

print("\nTotal features inspected:", len(contract))
print("Data contract saved as data_contract.json")
