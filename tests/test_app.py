
import pytest

from fastapi.testclient import TestClient

from app import app
from schemas import (
    AdmissionRequest,
    Semester1Request,
    DATA_CONTRACT
)

client = TestClient(app)


def create_payload(schema):
    """Generate valid test data using the data contract."""

    payload = {}

    for field in schema.model_fields.values():
        feature = field.alias
        details = DATA_CONTRACT[feature]

        if details["type"] in ("categorical", "ordinal"):
            payload[feature] = details["allowed_values"][0]

        else:
            minimum = details["minimum"]
            maximum = details["maximum"]

            payload[feature] = (minimum + maximum) / 2

    return payload


@pytest.fixture
def admission_payload():
    return create_payload(AdmissionRequest)


@pytest.fixture
def semester1_payload():
    return create_payload(Semester1Request)


# TEST 1: Home endpoint
def test_home_endpoint():
    response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Student Dropout Prediction" in response.text


# TEST 2: Health endpoint
def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["model_A"] == "loaded"
    assert response.json()["model_B"] == "loaded"


# TEST 3: Model A prediction
def test_admission_prediction(admission_payload):
    response = client.post(
        "/predict/admission",
        json=admission_payload
    )

    assert response.status_code == 200

    result = response.json()

    assert result["model"] == "Model A"
    assert result["prediction_stage"] == "Admission Time"
    assert result["prediction"] in [
        "Dropout", "Enrolled", "Graduate"
    ]

    probabilities = result["probabilities"]

    assert set(probabilities.keys()) == {
        "Dropout", "Enrolled", "Graduate"
    }

    assert sum(probabilities.values()) == pytest.approx(
        1.0, abs=0.0002
    )


# TEST 4: Model B prediction
def test_semester1_prediction(semester1_payload):
    response = client.post(
        "/predict/semester1",
        json=semester1_payload
    )

    assert response.status_code == 200

    result = response.json()

    assert result["model"] == "Model B"
    assert result["prediction_stage"] == "After First Semester"
    assert result["prediction"] in [
        "Dropout", "Enrolled", "Graduate"
    ]

    probabilities = result["probabilities"]

    assert set(probabilities.keys()) == {
        "Dropout", "Enrolled", "Graduate"
    }

    assert sum(probabilities.values()) == pytest.approx(
        1.0, abs=0.0002
    )


# TEST 5: Invalid age should return HTTP 422
def test_invalid_age(admission_payload):
    admission_payload["Age at enrollment"] = 100

    response = client.post(
        "/predict/admission",
        json=admission_payload
    )

    assert response.status_code == 422


# TEST 6: Invalid category should return HTTP 422
def test_invalid_category(admission_payload):
    admission_payload["Marital status"] = 99

    response = client.post(
        "/predict/admission",
        json=admission_payload
    )

    assert response.status_code == 422


# TEST 7: Domain logic
def test_stronger_student_has_higher_graduate_probability(
    admission_payload
):
    weaker_student = admission_payload.copy()
    stronger_student = admission_payload.copy()

    # Weaker academic profile
    weaker_student["Previous qualification (grade)"] = 100.0
    weaker_student["Admission grade"] = 100.0
    weaker_student["Age at enrollment"] = 25.0
    weaker_student["Debtor"] = 1
    weaker_student["Tuition fees up to date"] = 0

    # Stronger academic profile
    stronger_student["Previous qualification (grade)"] = 150.0
    stronger_student["Admission grade"] = 160.0
    stronger_student["Age at enrollment"] = 20.0
    stronger_student["Debtor"] = 0
    stronger_student["Tuition fees up to date"] = 1

    weaker_response = client.post(
        "/predict/admission",
        json=weaker_student
    )

    stronger_response = client.post(
        "/predict/admission",
        json=stronger_student
    )

    assert weaker_response.status_code == 200
    assert stronger_response.status_code == 200

    weaker_probability = (
        weaker_response.json()["probabilities"]["Graduate"]
    )

    stronger_probability = (
        stronger_response.json()["probabilities"]["Graduate"]
    )

    assert stronger_probability > weaker_probability
