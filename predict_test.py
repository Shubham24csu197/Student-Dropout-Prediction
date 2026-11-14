
import joblib
import pandas as pd

# Load cleaned dataset
df = pd.read_csv("data/student_dropout_cleaned.csv")

# Load trained models
model_A = joblib.load("model/model_A.joblib")
model_B = joblib.load("model/model_B.joblib")

# Select one actual student record
student_A = df[model_A.feature_names_in_].iloc[[0]]
student_B = df[model_B.feature_names_in_].iloc[[0]]

# Predict using Model A and Model B
prediction_A = model_A.predict(student_A)[0]
prediction_B = model_B.predict(student_B)[0]

# Prediction probabilities
probability_A = model_A.predict_proba(student_A)[0]
probability_B = model_B.predict_proba(student_B)[0]

print("MODEL A - Admission Time Prediction")
print("Predicted Outcome:", prediction_A)
print("Probabilities:", dict(zip(model_A.classes_, probability_A.round(3))))

print("\nMODEL B - After First Semester Prediction")
print("Predicted Outcome:", prediction_B)
print("Probabilities:", dict(zip(model_B.classes_, probability_B.round(3))))
