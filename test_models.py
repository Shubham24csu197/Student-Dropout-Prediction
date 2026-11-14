
import joblib
import os

# Model file paths
model_A_path = "model/model_A.joblib"
model_B_path = "model/model_B.joblib"

# Check whether files exist
assert os.path.exists(model_A_path), "Model A file not found!"
assert os.path.exists(model_B_path), "Model B file not found!"

# Load models
model_A = joblib.load(model_A_path)
model_B = joblib.load(model_B_path)

print("Model A loaded successfully!")
print("Model B loaded successfully!")

print("\nModel A Pipeline:")
print(model_A)

print("\nModel B Pipeline:")
print(model_B)

print("\nBoth models are ready for prediction.")
