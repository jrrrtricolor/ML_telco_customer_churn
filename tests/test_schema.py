import pandas as pd
import pytest
from pandera.errors import SchemaError

from src.schema import TELCO_COLUMNS, validate_telco_raw


def _valid_row() -> dict[str, object]:
    return {
        "customerID": "0001-TEST",
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 55.2,
        "TotalCharges": "662.4",
        "Churn": "No",
    }


def test_schema_valida_linha_telco_bruta():
    data = pd.DataFrame([_valid_row()], columns=TELCO_COLUMNS)

    validated = validate_telco_raw(data)

    assert list(validated.columns) == TELCO_COLUMNS


def test_schema_rejeita_target_invalido():
    row = _valid_row()
    row["Churn"] = "Maybe"
    data = pd.DataFrame([row], columns=TELCO_COLUMNS)

    with pytest.raises(SchemaError):
        validate_telco_raw(data)


def test_schema_valida_csv_local():
    data = pd.read_csv("data/raw/Telco_Customer_Churn.csv")

    validated = validate_telco_raw(data)

    assert len(validated) > 0
    assert set(validated["Churn"].unique()).issubset({"Yes", "No"})
