
import json
from typing import Literal, Annotated

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    create_model
)

# Load the inspected dataset contract
with open("data_contract.json", "r") as file:
    DATA_CONTRACT = json.load(file)


# Feature groups
CATEGORICAL_FEATURES = [
    "Marital status",
    "Application mode",
    "Course",
    "Previous qualification",
    "Nationality",
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
    "Daytime/evening attendance",
    "Displaced",
    "Educational special needs",
    "Debtor",
    "Tuition fees up to date",
    "Gender",
    "Scholarship holder",
    "International",
]

ORDINAL_FEATURES = ["Application order"]

NUMERICAL_FEATURES = [
    "Previous qualification (grade)",
    "Admission grade",
    "Age at enrollment",
    "Unemployment rate",
    "Inflation rate",
    "GDP",
    "Curricular units 1st sem (credited)",
    "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (evaluations)",
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)",
    "Curricular units 1st sem (without evaluations)",
]


def make_field_name(name):
    return (
        name.lower()
        .replace("'", "")
        .replace("(", "")
        .replace(")", "")
        .replace("/", "_")
        .replace(" ", "_")
    )


def create_request_schema(schema_name, features):
    fields = {}

    for feature in features:
        details = DATA_CONTRACT[feature]

        if details["type"] in ("categorical", "ordinal"):
            # Only allow values observed in the dataset
            allowed = tuple(details["allowed_values"])
            field_type = Literal[allowed]

        else:
            # Enforce observed numerical minimum and maximum
            field_type = Annotated[
                float,
                Field(
                    ge=details["minimum"],
                    le=details["maximum"]
                )
            ]

        fields[make_field_name(feature)] = (
            field_type,
            Field(..., alias=feature)
        )

    return create_model(
        schema_name,
        __config__=ConfigDict(
            extra="forbid",
            populate_by_name=True
        ),
        **fields
    )


# Model A: admission-time prediction (24 features)
AdmissionRequest = create_request_schema(
    "AdmissionRequest",
    CATEGORICAL_FEATURES
    + ORDINAL_FEATURES
    + NUMERICAL_FEATURES[:6]
)

# Model B: after first semester (30 features)
Semester1Request = create_request_schema(
    "Semester1Request",
    CATEGORICAL_FEATURES
    + ORDINAL_FEATURES
    + NUMERICAL_FEATURES
)
