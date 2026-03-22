from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api import app

CLIENT = TestClient(app)
CSV_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "teste-2-variaveis_explicaveis.csv"

# Minimum accuracy the model must achieve on the sampled rows.
MIN_ACCURACY = 0.75


def _row_to_payload(row: pd.Series) -> dict:
    return {
        "gender": row["gender"],
        "SeniorCitizen": int(row["SeniorCitizen"]),
        "Partner": row["Partner"],
        "Dependents": row["Dependents"],
        "tenure": int(row["tenure"]),
        "PhoneService": row["PhoneService"],
        "MultipleLines": row["MultipleLines"],
        "InternetService": row["InternetService"],
        "OnlineSecurity": row["OnlineSecurity"],
        "OnlineBackup": row["OnlineBackup"],
        "DeviceProtection": row["DeviceProtection"],
        "TechSupport": row["TechSupport"],
        "StreamingTV": row["StreamingTV"],
        "StreamingMovies": row["StreamingMovies"],
        "Contract": row["Contract"],
        "PaperlessBilling": row["PaperlessBilling"],
        "PaymentMethod": row["PaymentMethod"],
        "MonthlyCharges": float(row["MonthlyCharges"]),
        "TotalCharges": float(row["TotalCharges"]),
    }


@pytest.fixture(scope="module")
def csv_data() -> pd.DataFrame:
    return pd.read_csv(CSV_PATH)


def test_health():
    response = CLIENT.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_returns_list(csv_data):
    row = csv_data.iloc[0]
    response = CLIENT.post("/predict", json=_row_to_payload(row))

    assert response.status_code == 200
    assert isinstance(response.json()["prediction"], list)


def test_predict_response_is_binary(csv_data):
    """Each prediction must be 0 (no churn) or 1 (churn)."""
    for idx in range(10):
        row = csv_data.iloc[idx]
        response = CLIENT.post("/predict", json=_row_to_payload(row))
        assert response.status_code == 200
        assert response.json()["prediction"][0] in (0, 1)


def test_predict_response_has_churn(csv_data):
    """Should have some 1 (churn) predictions."""
    predictions = 0

    for idx in range(500):
        row = csv_data.iloc[idx]
        response = CLIENT.post("/predict", json=_row_to_payload(row))
        assert response.status_code == 200
        assert response.json()["prediction"][0] in (0, 1)
        if response.json()["prediction"][0] == 1:
            predictions += 1

    assert predictions > 0, "No churn predictions found in the first 500 rows."

def test_predict_accuracy_on_sample(csv_data):
    """Model must reach MIN_ACCURACY on the first 50 rows of the test file."""
    sample = csv_data.head(50)
    correct = 0

    for _, row in sample.iterrows():
        expected = int(row["Churn"])
        response = CLIENT.post("/predict", json=_row_to_payload(row))
        assert response.status_code == 200
        predicted = response.json()["prediction"][0]
        if predicted == expected:
            correct += 1

    accuracy = correct / len(sample)
    assert accuracy >= MIN_ACCURACY, (
        f"Accuracy {accuracy:.2%} is below the minimum threshold of {MIN_ACCURACY:.0%}"
    )


def test_predict_invalid_enum_returns_422(csv_data):
    row = csv_data.iloc[0]
    payload = _row_to_payload(row)
    payload["gender"] = "Unknown"

    response = CLIENT.post("/predict", json=payload)

    assert response.status_code == 422
