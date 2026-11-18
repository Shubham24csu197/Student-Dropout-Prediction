# Student Dropout Prediction (CSL236 End-Term Project)

A production-style machine learning system that predicts student academic outcomes (`Dropout`, `Enrolled`, `Graduate`) at two critical institutional milestones: **admission time** and **after the first semester**.

Built following the **CSL236 End-Term Project Submission Guidelines** and reference MLOps architecture at The NorthCap University.

---

## 1. Project Overview & Problem Statement

Student attrition is one of the most significant challenges confronting higher education institutions. Early identification of vulnerable students allows academic advisors to intervene before attrition becomes irreversible.

This project implements an end-to-end ML lifecycle:
- **Data Contract & Pydantic Validation**: Rigorous validation returning HTTP 422 for out-of-bound inputs.
- **Experiment Tracking (MLflow)**: Multiple runs and algorithm comparisons tracked in an SQLite backend (`sqlite:///mlflow.db`).
- **Validated FastAPI Service**: Serves predictions with probability distributions and 2–3 actionable, rule-based tips.
- **Automated Testing**: 8 Pytest suites covering unit, boundary, and domain logic.
- **Lean Containerization**: Built from `python:3.11-slim` using `requirements-serve.txt` (~210 MB).
- **CI/CD Automation**: GitHub Actions pipeline triggering automated training and testing on push to main.
- **Drift Monitoring**: Continuous logging of inference payloads and automated drift detection script (`drift_check.py`).

---

## 2. Live Application & Endpoints

- **Live Web Application**: [https://student-dropout-prediction-4u8u.onrender.com](https://student-dropout-prediction-4u8u.onrender.com)
- **Interactive API Documentation (Swagger)**: [https://student-dropout-prediction-4u8u.onrender.com/docs](https://student-dropout-prediction-4u8u.onrender.com/docs)
- **Health Check Endpoint**: [https://student-dropout-prediction-4u8u.onrender.com/health](https://student-dropout-prediction-4u8u.onrender.com/health)

---

## 3. Dataset & Mandatory Data Contract

- **Dataset**: *Predict Students' Dropout and Academic Success*
- **Source**: UCI Machine Learning Repository (Dataset ID: 697)  
  [https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success)
- **Size**: 4,424 student records, 36 features, 3 target classes (stored in `data/student_dropout.csv`).

### Data Contract Table

Every feature constraint in this table is enforced in `schemas.py`. Any out-of-range or wrongly typed input returns HTTP 422.

| Feature | Type | Allowed range / values | Why it matters |
|---|---|---|---|
| Marital status | categorical | 1, 2, 3, 4, 5, 6 | Indicator of personal and family stability |
| Application mode | categorical | 1, 2, 5, 7, 10, 15, 16, 17, 18, 26, 27, 39, 42, 43, 44, 51, 53, 57 | Admission channel (e.g., general quota, transfer, mature) |
| Application order | ordinal | 0 – 9 | Preference ranking of the course at application time |
| Course | categorical | 33, 171, 8014, 9003, 9070, 9085, 9119, 9130, 9147, 9238, 9254, 9500, 9556, 9670, 9773, 9853, 9991 | Program of study (degree vs. vocational diploma) |
| Daytime/evening attendance | categorical | {0, 1} | Work-study balance indicator (1 = Daytime, 0 = Evening) |
| Previous qualification | categorical | 1, 2, 3, 4, 5, 6, 9, 10, 12, 14, 15, 19, 38, 39, 40, 42, 43 | Educational background prior to university entry |
| Previous qualification (grade) | float | 95.0 – 190.0 | Academic baseline and prior preparation |
| Nationality | categorical | 1, 2, 6, 11, 13, 14, 17, 21, 22, 24, 25, 26, 32, 41, 62, 100, 101, 103, 105, 108, 109 | Domestic vs international student background |
| Mother's qualification | categorical | 1 – 44 | Socioeconomic and parental educational attainment |
| Father's qualification | categorical | 1 – 44 | Socioeconomic and parental educational attainment |
| Mother's occupation | categorical | 0 – 194 | Household employment stability and parental support |
| Father's occupation | categorical | 0 – 195 | Household employment stability and parental support |
| Admission grade | float | 95.0 – 190.0 | Competitive entrance examination score |
| Displaced | categorical | {0, 1} | Living away from hometown (relocation adjustment factor) |
| Educational special needs | categorical | {0, 1} | Requirement for specialized educational accommodations |
| Debtor | categorical | {0, 1} | Existing overdue balance to the university (strong dropout risk signal) |
| Tuition fees up to date | categorical | {0, 1} | Financial standing with the institution |
| Gender | categorical | {0, 1} | Demographic factor (1 = Male, 0 = Female) |
| Scholarship holder | categorical | {0, 1} | Institutional financial aid backing |
| Age at enrollment | float | 17.0 – 70.0 | Mature student indicator versus traditional direct entry |
| International | categorical | {0, 1} | Non-native cultural, linguistic, and regulatory context |
| Curricular units 1st sem (credited) | float | 0.0 – 20.0 | Prior learning recognition and transfer credit |
| Curricular units 1st sem (enrolled) | float | 0.0 – 26.0 | Academic study workload registered |
| Curricular units 1st sem (evaluations) | float | 0.0 – 45.0 | Examination engagement and exam attendance |
| Curricular units 1st sem (approved) | float | 0.0 – 26.0 | Direct metric of academic competency in semester 1 |
| Curricular units 1st sem (grade) | float | 0.0 – 18.88 | Grade point average for semester 1 (scale 0–20) |
| Curricular units 1st sem (without evaluations) | float | 0.0 – 12.0 | Courses skipped/abandoned (strong absenteeism indicator) |
| Unemployment rate | float | 7.6 – 16.2 | Macroeconomic environment and local job scarcity |
| Inflation rate | float | -0.8 – 3.7 | Cost of living pressure on students |
| GDP | float | -4.06 – 3.51 | Regional economic growth indicator |
| **Target** | categorical | **{Dropout, Enrolled, Graduate}** | **Ground-truth academic outcome (Prediction Label)** |

---

## 4. Experiment Tracking & MLflow Results

Experiments are tracked using MLflow with SQLite backend (`sqlite:///mlflow.db`) as required by Stage 2. We trained and evaluated multiple runs comparing **Logistic Regression** and **Random Forest**.

### MLflow Experiment Results Table

| Run Name | Algorithm | Stage / Feature Count | Key Hyperparameters | Test Accuracy | Macro F1 | Status |
|---|---|---|---|---:|---:|---|
| `Model_A_LogisticRegression` | Logistic Regression | Admission Time (24 features) | `C=1.0`, `class_weight='balanced'`, `max_iter=1000` | 0.5706 | 0.5465 | Baseline Admission |
| `Model_B_LogisticRegression` | Logistic Regression | After Semester 1 (30 features) | `C=1.0`, `class_weight='balanced'`, `max_iter=1000` | 0.6983 | 0.6585 | Primary Production Model |
| `Model_B_RandomForest_n100_d6` | Random Forest | After Semester 1 (30 features) | `n_estimators=100`, `max_depth=6` | 0.7040 | 0.6557 | Comparison Algorithm |
| `Model_B_RandomForest_n150_d8` | Random Forest | After Semester 1 (30 features) | `n_estimators=150`, `max_depth=8` | 0.7073 | 0.6613 | Tuned RF |
| `Model_B_RandomForest_Tuned_n150_d10` | Random Forest | After Semester 1 (30 features) | `n_estimators=150`, `max_depth=10` | **0.7232** | **0.6769** | Best Macro F1 & Accuracy |

### Model Selection Justification
- **Model B (First Semester)** substantially outperforms Model A (Accuracy increases from 57.06% to 72.32%, Macro F1 increases from 0.5465 to 0.6769), showing that early semester 1 academic engagement (approved units and evaluation rate) is the strongest determinant of graduation vs dropout.
- **Selected Serving Model**: `Model_B_LogisticRegression` with class weighting was selected for production serving because of its excellent probability calibration, sub-millisecond inference latency, and high interpretability, with `Model_B_RandomForest` providing the competitive benchmark logged in MLflow. Both are serialized and accessible.

---

## 5. Repository Structure

```text
Student-Dropout-Prediction/
├── data/
│   ├── student_dropout.csv             # Full cleaned dataset (4,424 records)
│   └── student_dropout_cleaned.csv
├── model/
│   ├── model.joblib                    # Serialized best model (Stage 3)
│   ├── model_A.joblib                  # Admission-time pipeline
│   └── model_B.joblib                  # First-semester pipeline
├── static/
│   └── index.html                      # Interactive web UI with rule-based tips
├── tests/
│   └── test_app.py                     # 8 Pytest suites (health, UI, domain logic, 422)
├── logs/
│   └── prediction_inputs.log           # Logged inference inputs for drift monitoring
├── .github/workflows/
│   └── ci.yml                          # GitHub Actions CI (install, train, test)
├── app.py                              # FastAPI service + logging + tips
├── schemas.py                          # Strict Pydantic validation schemas
├── data_contract.json                  # Machine-readable data contract
├── train.py                            # CLI training script with MLflow tracking
├── drift_check.py                      # Bonus: Data drift monitoring script
├── Dockerfile                          # Serving container (python:3.11-slim)
├── requirements.txt                    # Full dev/training environment
├── requirements-serve.txt              # Lean serving runtime (no MLflow/pytest)
├── .gitignore                          # Excludes venv/, __pycache__/, keeps mlflow.db
└── README.md                           # Documentation and submission checklist
```

---

## 6. Setup and Run Instructions

### Step 1: Environment Setup
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Step 2: Train Model & Track in MLflow
```powershell
python train.py --n_estimators 100 --max_depth 6
```
To run full hyperparameter sweep across all algorithms:
```powershell
python train.py --all_runs
```

### Step 3: Open MLflow Experiment Dashboard
```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
Open [http://127.0.0.1:5000](http://127.0.0.1:5000) to inspect tracked runs, metrics, and hyperparameter comparisons.

### Step 4: Run Automated Tests
```powershell
pytest -v --tb=short
```

### Step 5: Start the API & UI
```powershell
uvicorn app:app --reload --port 8000
```
- Web Application: [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
- Swagger API Docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- Health Status: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

### Step 6: Docker Deployment
```powershell
docker build -t student-dropout-api:v1 .
docker run -d --name student-dropout-app -p 8000:8000 --restart unless-stopped student-dropout-api:v1
```
- **Docker Image Base**: `python:3.11-slim`
- **Docker Image Size**: **~210 MB** (measured via `docker images`)

---

## 7. Data Drift Monitoring (Bonus: +5 Marks)

Every prediction sent to `/predict`, `/predict/admission`, or `/predict/semester1` is automatically logged with an ISO timestamp into `logs/prediction_inputs.log`.

Run the automated drift detection script:
```powershell
python drift_check.py
```
This script computes mean and standard deviation for live incoming features versus training distributions and alerts when distribution shifts occur.

---

## 8. Authors & Team Details

| Name | Roll Number | Individual Contribution |
|---|---|---|
| **Shubham** | *24CSU197* | Data preprocessing, feature engineering, Model A & B training, MLflow tracking integration. |
| **Sarthak** | *24CSUxxx* | FastAPI architecture, Pydantic data contract enforcement, Pytest suite, Dockerization, UI integration. |

---

## 9. Submission Checklist

- [x] Repository is public on GitHub
- [x] `Fresh clone + pip install -r requirements.txt + python train.py` works seamlessly
- [x] All 8 Pytest tests pass locally and in GitHub Actions CI
- [x] Docker image builds from `python:3.11-slim` using `requirements-serve.txt` (~210 MB)
- [x] Live URL deployed on Render with working predictions and UI
- [x] Invalid inputs return HTTP 422 with validation errors
- [x] MLflow runs tracked in SQLite (`sqlite:///mlflow.db`) and committed
- [x] Live prediction logging enabled for drift checks (`drift_check.py`)
- [x] 2–3 actionable rule-based tips provided in API and UI
