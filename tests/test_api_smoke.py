import importlib
import sys

import numpy as np
from fastapi.testclient import TestClient


class FakeModel:
    def predict(self, data):
        return np.array([1])

    def predict_proba(self, data):
        return np.array([[0.2, 0.8]])


def _payload() -> dict[str, object]:
    return {
        "gender": "Male",
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
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 55.2,
        "TotalCharges": 662.4,
    }


def test_api_health_predict_smoke(monkeypatch):
    import mlflow.sklearn

    monkeypatch.setattr(mlflow.sklearn, "load_model", lambda _uri: FakeModel())
    sys.modules.pop("src.api", None)
    api = importlib.import_module("src.api")

    client = TestClient(api.app)

    health = client.get("/health")
    prediction = client.post("/predict", json=_payload())

    assert health.status_code == 200
    assert health.json() == {"status": "ok"}
    assert prediction.status_code == 200
    assert prediction.json() == {"prediction": True}
