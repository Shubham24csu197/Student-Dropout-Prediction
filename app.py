
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

from schemas import AdmissionRequest, Semester1Request

app = FastAPI(
    title="Student Dropout Prediction API",
    description="Predict student academic outcomes using Machine Learning",
    version="1.0.0"
)

# Load trained models
model_A = joblib.load("model/model_A.joblib")
model_B = joblib.load("model/model_B.joblib")

TARGET_NAMES = ["Dropout", "Enrolled", "Graduate"]


@app.get("/")
def home():
    return {
        "message": "Student Dropout Prediction API is running!",
        "models": ["Model A", "Model B"]
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_A": "loaded",
        "model_B": "loaded"
    }


def generate_prediction(model, request_data):
    # Convert validated request into a DataFrame
    input_df = pd.DataFrame([request_data])

    # Predict class
    prediction = model.predict(input_df)[0]

    # Predict class probabilities
    probabilities = model.predict_proba(input_df)[0]

    return {
        "prediction": str(prediction),
        "probabilities": {
            str(label): round(float(probability), 4)
            for label, probability in zip(
                model.classes_,
                probabilities
            )
        }
    }


@app.post("/predict/admission")
def predict_admission(data: AdmissionRequest):
    request_data = data.model_dump(by_alias=True)

    result = generate_prediction(model_A, request_data)

    return {
        "prediction_stage": "Admission Time",
        "model": "Model A",
        **result
    }


@app.post("/predict/semester1")
def predict_semester1(data: Semester1Request):
    request_data = data.model_dump(by_alias=True)

    result = generate_prediction(model_B, request_data)

    return {
        "prediction_stage": "After First Semester",
        "model": "Model B",
        **result
    }
