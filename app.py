import json
from datetime import datetime
from pathlib import Path
from typing import Union, List

import joblib
import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from document_upload import prepare_preview, ADMISSION_FEATURES, SEMESTER_FEATURES

from schemas import AdmissionRequest, Semester1Request

app = FastAPI(
    title="Student Dropout Prediction API",
    description="Predict student academic outcomes using Machine Learning with MLflow tracking",
    version="1.0.0",
)

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
LOGS_DIR = BASE_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOGS_DIR / "prediction_inputs.log"

# Load trained models once at application startup (CSL236 Stage 3)
model_file = BASE_DIR / "model" / "model.joblib"
model_A_file = BASE_DIR / "model" / "model_A.joblib"
model_B_file = BASE_DIR / "model" / "model_B.joblib"

# Load serialized models
model_A = joblib.load(model_A_file if model_A_file.exists() else model_file)
model_B = joblib.load(model_B_file if model_B_file.exists() else model_file)
model_primary = joblib.load(model_file if model_file.exists() else model_B_file)

TARGET_NAMES = ["Dropout", "Enrolled", "Graduate"]

# Mount static frontend
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def log_prediction_input(endpoint: str, request_data: dict, prediction: str):
    """Log prediction inputs to file for drift monitoring (CSL236 Stage 8)."""
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "endpoint": endpoint,
        "prediction": prediction,
        "input_features": request_data,
    }
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception as e:
        print(f"Failed to log prediction input: {e}")


def get_rule_based_tips(prediction: str, request_data: dict) -> List[str]:
    """Provide 2-3 rule-based tips for the student (CSL236 Stage 4)."""
    tips = []
    if prediction == "Dropout":
        tips.append(
            "High risk detected: Schedule an academic advising and counseling session immediately."
        )
        if (
            request_data.get("Tuition fees up to date") == 0
            or request_data.get("Debtor") == 1
        ):
            tips.append(
                "Financial warning: Connect with the financial aid office to review tuition payment plans or scholarship opportunities."
            )
        else:
            tips.append(
                "Academic alert: Review prerequisite syllabus and enroll in peer tutoring for challenging courses."
            )
        if request_data.get(
            "Curricular units 1st sem (approved)", 0
        ) < request_data.get("Curricular units 1st sem (enrolled)", 1):
            tips.append(
                "Credit deficit: Create a recovery plan to clear unapproved curricular units next semester."
            )
        else:
            tips.append(
                "Attendance monitoring: Maintain attendance above 85% in all registered courses."
            )
    elif prediction == "Enrolled":
        tips.append(
            "Borderline academic outcome: Closely track 2nd semester attendance and internal assessments."
        )
        tips.append(
            "Coursework focus: Form study groups to boost GPA and clear any pending evaluations."
        )
        tips.append(
            "Faculty mentoring: Meet your course coordinator to discuss academic milestones."
        )
    else:  # Graduate
        tips.append(
            "Strong academic trajectory: Maintain current study habits and academic performance."
        )
        tips.append(
            "Career readiness: Engage with the campus placement cell for internship and project opportunities."
        )
        tips.append(
            "Skill development: Consider taking advanced electives or certifications in your domain."
        )
    return tips[:3]


def generate_prediction(model, request_data):
    input_df = pd.DataFrame([request_data])
    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]

    return {
        "prediction": str(prediction),
        "probabilities": {
            str(label): round(float(probability), 4)
            for label, probability in zip(model.classes_, probabilities)
        },
    }


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "loaded",
        "model_A": "loaded",
        "model_B": "loaded",
        "logging": "enabled",
    }


@app.post("/predict")
def predict(data: Union[Semester1Request, AdmissionRequest]):
    """Standard POST /predict endpoint required by CSL236 guidelines."""
    request_data = data.model_dump(by_alias=True)
    if "Curricular units 1st sem (credited)" in request_data:
        result = generate_prediction(model_primary, request_data)
        stage = "After First Semester"
        model_name = "Model B (Primary)"
    else:
        result = generate_prediction(model_A, request_data)
        stage = "Admission Time"
        model_name = "Model A"

    log_prediction_input("/predict", request_data, result["prediction"])
    tips = get_rule_based_tips(result["prediction"], request_data)

    return {"prediction_stage": stage, "model": model_name, "tips": tips, **result}


@app.post("/predict/admission")
def predict_admission(data: AdmissionRequest):
    request_data = data.model_dump(by_alias=True)
    result = generate_prediction(model_A, request_data)
    log_prediction_input("/predict/admission", request_data, result["prediction"])
    tips = get_rule_based_tips(result["prediction"], request_data)

    return {
        "prediction_stage": "Admission Time",
        "model": "Model A",
        "tips": tips,
        **result,
    }


@app.post("/predict/semester1")
def predict_semester1(data: Semester1Request):
    request_data = data.model_dump(by_alias=True)
    result = generate_prediction(model_B, request_data)
    log_prediction_input("/predict/semester1", request_data, result["prediction"])
    tips = get_rule_based_tips(result["prediction"], request_data)

    return {
        "prediction_stage": "After First Semester",
        "model": "Model B",
        "tips": tips,
        **result,
    }


@app.post("/upload/preview")
async def upload_preview(file: UploadFile = File(...), model_name: str = Form(...)):
    if model_name not in ("Model A", "Model B"):
        raise HTTPException(
            status_code=400, detail="model_name must be Model A or Model B"
        )
    content = await file.read()
    return prepare_preview(file.filename or "upload", content, model_name)


@app.post("/upload/predict")
def upload_predict(payload: dict):
    model_name = payload.get("model")
    features = payload.get("features")
    if model_name not in ("Model A", "Model B") or not isinstance(features, dict):
        raise HTTPException(
            status_code=422, detail="Provide model and extracted features"
        )
    schema = AdmissionRequest if model_name == "Model A" else Semester1Request
    try:
        validated = schema.model_validate(features)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    request_data = validated.model_dump(by_alias=True)
    model = model_A if model_name == "Model A" else model_B
    result = generate_prediction(model, request_data)
    return {
        "prediction_stage": (
            "Admission Time" if model_name == "Model A" else "After First Semester"
        ),
        "model": model_name,
        **result,
    }
