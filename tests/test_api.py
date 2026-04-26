from fastapi.testclient import TestClient

import src.api as api


class DummyModel:
    def predict(self, data):
        return [int(data["MonthlyCharges"].iloc[0] > 50)]


def _payload_valido() -> dict:
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


def test_health_retorna_status_ok():
    client = TestClient(api.app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_retorna_booleano_sem_carregar_mlflow(monkeypatch):
    monkeypatch.setattr(api, "_MODEL", DummyModel())
    client = TestClient(api.app)

    response = client.post("/predict", json=_payload_valido())

    assert response.status_code == 200
    assert isinstance(response.json()["prediction"], bool)
