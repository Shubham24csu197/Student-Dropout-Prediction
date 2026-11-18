from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from fastapi import UploadFile, File, Form, HTTPException
from document_upload import prepare_preview, ADMISSION_FEATURES, SEMESTER_FEATURES
import joblib
import pandas as pd
from pathlib import Path

from schemas import AdmissionRequest, Semester1Request

app = FastAPI(
    title="Student Dropout Prediction API",
    description="Predict student academic outcomes using Machine Learning",
    version="1.0.0"
)

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

# Load trained models
model_A = joblib.load(BASE_DIR / "model" / "model_A.joblib")
model_B = joblib.load(BASE_DIR / "model" / "model_B.joblib")

TARGET_NAMES = ["Dropout", "Enrolled", "Graduate"]

# Serve CSS/JS or other static assets from /static
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_A": "loaded",
        "model_B": "loaded"
    }


def generate_prediction(model, request_data):
    input_df = pd.DataFrame([request_data])
    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]

    return {
        "prediction": str(prediction),
        "probabilities": {
            str(label): round(float(probability), 4)
            for label, probability in zip(model.classes_, probabilities)
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


@app.post("/upload/preview")
async def upload_preview(file: UploadFile = File(...), model_name: str = Form(...)):
    if model_name not in ("Model A", "Model B"):
        raise HTTPException(status_code=400, detail="model_name must be Model A or Model B")
    content = await file.read()
    return prepare_preview(file.filename or "upload", content, model_name)


@app.post("/upload/predict")
def upload_predict(payload: dict):
    model_name = payload.get("model")
    features = payload.get("features")
    if model_name not in ("Model A", "Model B") or not isinstance(features, dict):
        raise HTTPException(status_code=422, detail="Provide model and extracted features")
    schema = AdmissionRequest if model_name == "Model A" else Semester1Request
    try:
        validated = schema.model_validate(features)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    request_data = validated.model_dump(by_alias=True)
    model = model_A if model_name == "Model A" else model_B
    result = generate_prediction(model, request_data)
    return {"prediction_stage": "Admission Time" if model_name == "Model A" else "After First Semester", "model": model_name, **result}
