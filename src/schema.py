import pandera.pandas as pa
from pandera.typing import DataFrame

TELCO_COLUMNS = [
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
]


TELCO_RAW_SCHEMA = pa.DataFrameSchema(
    {
        "customerID": pa.Column(str, nullable=False),
        "gender": pa.Column(str, checks=pa.Check.isin(["Female", "Male"])),
        "SeniorCitizen": pa.Column(int, checks=pa.Check.isin([0, 1])),
        "Partner": pa.Column(str, checks=pa.Check.isin(["Yes", "No"])),
        "Dependents": pa.Column(str, checks=pa.Check.isin(["Yes", "No"])),
        "tenure": pa.Column(int, checks=pa.Check.ge(0)),
        "PhoneService": pa.Column(str, checks=pa.Check.isin(["Yes", "No"])),
        "MultipleLines": pa.Column(
            str,
            checks=pa.Check.isin(["Yes", "No", "No phone service"]),
        ),
        "InternetService": pa.Column(
            str,
            checks=pa.Check.isin(["DSL", "Fiber optic", "No"]),
        ),
        "OnlineSecurity": pa.Column(
            str,
            checks=pa.Check.isin(["Yes", "No", "No internet service"]),
        ),
        "OnlineBackup": pa.Column(
            str,
            checks=pa.Check.isin(["Yes", "No", "No internet service"]),
        ),
        "DeviceProtection": pa.Column(
            str,
            checks=pa.Check.isin(["Yes", "No", "No internet service"]),
        ),
        "TechSupport": pa.Column(
            str,
            checks=pa.Check.isin(["Yes", "No", "No internet service"]),
        ),
        "StreamingTV": pa.Column(
            str,
            checks=pa.Check.isin(["Yes", "No", "No internet service"]),
        ),
        "StreamingMovies": pa.Column(
            str,
            checks=pa.Check.isin(["Yes", "No", "No internet service"]),
        ),
        "Contract": pa.Column(
            str,
            checks=pa.Check.isin(["Month-to-month", "One year", "Two year"]),
        ),
        "PaperlessBilling": pa.Column(str, checks=pa.Check.isin(["Yes", "No"])),
        "PaymentMethod": pa.Column(
            str,
            checks=pa.Check.isin(
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ]
            ),
        ),
        "MonthlyCharges": pa.Column(float, checks=pa.Check.gt(0)),
        "TotalCharges": pa.Column(str, nullable=False),
        "Churn": pa.Column(str, checks=pa.Check.isin(["Yes", "No"])),
    },
    strict=True,
    coerce=True,
)


def validate_telco_raw(data: DataFrame) -> DataFrame:
    return TELCO_RAW_SCHEMA.validate(data)
