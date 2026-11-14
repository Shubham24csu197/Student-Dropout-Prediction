
# Student Dropout Prediction

A machine-learning-based system that predicts student academic outcomes using information available at two different stages: admission time and after the first semester.

## Live Application

**Website:** https://student-dropout-prediction-4u8u.onrender.com

**API Documentation:** https://student-dropout-prediction-4u8u.onrender.com/docs

## Project Overview

Student dropout is an important challenge for educational institutions. Identifying students who may be at risk can help institutions provide timely academic support.

This project uses machine learning to classify students into three possible academic outcomes:

- Dropout
- Enrolled
- Graduate

The system provides predictions at two stages:

- **Model A – Admission-Time Prediction:** Uses 24 features available at the time of admission.
- **Model B – First-Semester Prediction:** Uses admission information along with 6 additional first-semester academic indicators.

The predictions are intended for decision support, not as guaranteed outcomes.

## Dataset

**Dataset:** Predict Students' Dropout and Academic Success

**Source:** UCI Machine Learning Repository

https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success

Dataset characteristics:

- 4,424 student records
- 36 input features
- 3 target classes
- No missing values in the processed dataset

The dataset comes from one Portuguese higher education institution. Results may not generalize directly to other institutions or countries.

## Machine Learning

The project compares Logistic Regression and Random Forest models.

The selected models are Logistic Regression pipelines:

| Model | Input features | Purpose |
|---|---:|---|
| Model A | 24 | Admission-time prediction |
| Model B | 30 | Prediction after first semester |

Model B includes six additional first-semester indicators.

### Preprocessing

- One-hot encoding for categorical features
- Standard scaling for numerical features
- Stratified train-test split
- Class-weight balancing for the selected models

### Evaluation

Models were evaluated using:

- Accuracy
- Precision
- Recall
- Macro F1-score
- Confusion matrix
- Cross-validation

Macro F1-score was considered during model selection because the target classes are imbalanced.

## Technology Stack

- Python 3.11
- Pandas and NumPy
- Scikit-learn
- MLflow
- FastAPI
- Pydantic
- HTML, CSS and JavaScript
- Docker
- Git and GitHub
- GitHub Actions
- Render

## Project Structure

```text
Student-Dropout-Prediction/
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── model/
│   ├── model_A.joblib
│   └── model_B.joblib
│
├── static/
│   └── index.html
│
├── tests/
│   └── test_app.py
│
├── app.py
├── schemas.py
├── data_contract.json
├── Dockerfile
├── .dockerignore
├── .gitignore
├── requirements-serve.txt
├── manual_predict.py
└── README.md
```

## Running Locally

### 1. Clone the repository

```bash
git clone https://github.com/Shubham24csu197/Student-Dropout-Prediction.git
cd Student-Dropout-Prediction
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the environment

Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
pip install -r requirements-serve.txt
```

### 5. Start the API

```bash
uvicorn app:app --reload
```

Open the following URLs:

- Website: http://127.0.0.1:8000/
- API documentation: http://127.0.0.1:8000/docs
- Health check: http://127.0.0.1:8000/health

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Application homepage |
| GET | `/health` | API and model health status |
| POST | `/predict/admission` | Admission-time prediction |
| POST | `/predict/semester1` | First-semester prediction |

The API uses Pydantic validation to check incoming input values. Invalid inputs return HTTP 422.

## Docker

Build the Docker image:

```bash
docker build -t student-dropout-api:v1 .
```

Run the container:

```bash
docker run -d --name student-dropout-container -p 8000:8000 student-dropout-api:v1
```

## Testing and CI

The project includes automated API tests using Pytest and FastAPI TestClient.

The tests cover:

- Homepage availability
- Health endpoint
- Admission-time prediction
- First-semester prediction
- Invalid numerical input
- Invalid categorical input
- Basic prediction logic

GitHub Actions runs the automated tests on repository pushes and pull requests.

## Limitations

- The dataset represents one Portuguese higher education institution.
- Predictions depend on the quality and distribution of the training data.
- Model probabilities are estimates, not certainties.
- The model should be validated on institution-specific data before real-world use.
- The system is intended to support academic intervention, not replace human decisions.

## Future Improvements

- Validate the models using data from multiple institutions.
- Improve prediction performance for the Enrolled class.
- Add explainability using SHAP.
- Add student-level risk monitoring and dashboards.
- Monitor model performance and data drift after deployment.

## Authors

**Shubham **  
**Sarthak**

The project was developed as part of the CSL236 – Introduction to AI & ML course at The NorthCap University.