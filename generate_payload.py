
import pandas as pd
import json

from schemas import AdmissionRequest, Semester1Request

# Load the cleaned dataset
df = pd.read_csv("data/student_dropout_cleaned.csv")

# Select the first student record
student = df.iloc[0]


def create_payload(schema):
    payload = {}

    for field in schema.model_fields.values():
        original_name = field.alias
        value = student[original_name]

        if field.annotation == int:
            payload[original_name] = int(value)
        else:
            payload[original_name] = float(value)

    return payload


admission_payload = create_payload(AdmissionRequest)
semester1_payload = create_payload(Semester1Request)

print("\n========== MODEL A: ADMISSION INPUT ==========")
print(json.dumps(admission_payload, indent=2))

print("\n========== MODEL B: SEMESTER 1 INPUT ==========")
print(json.dumps(semester1_payload, indent=2))
