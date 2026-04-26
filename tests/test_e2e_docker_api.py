import subprocess
import time

import pytest
import requests

IMAGE_TAG = "churn-api:e2e-test"
CONTAINER_NAME = "churn-api-e2e-test"
BASE_URL = "http://127.0.0.1:8010"


def _build_payload(i: int) -> dict:
    # Gera dados simples e variados para evitar 1000 chamadas idênticas.
    return {
        "gender": "Male" if i % 2 == 0 else "Female",
        "SeniorCitizen": i % 2,
        "Partner": "Yes" if i % 3 == 0 else "No",
        "Dependents": "No" if i % 4 == 0 else "Yes",
        "tenure": 1 + (i % 72),
        "PhoneService": "Yes",
        "MultipleLines": "Yes" if i % 2 == 0 else "No",
        "InternetService": ["DSL", "Fiber optic", "No"][i % 3],
        "OnlineSecurity": ["Yes", "No", "No internet service"][i % 3],
        "OnlineBackup": ["No", "Yes", "No internet service"][(i + 1) % 3],
        "DeviceProtection": ["No", "Yes", "No internet service"][(i + 2) % 3],
        "TechSupport": ["No", "Yes", "No internet service"][i % 3],
        "StreamingTV": ["Yes", "No", "No internet service"][(i + 1) % 3],
        "StreamingMovies": ["Yes", "No", "No internet service"][(i + 2) % 3],
        "Contract": ["Month-to-month", "One year", "Two year"][i % 3],
        "PaperlessBilling": "Yes" if i % 2 == 0 else "No",
        "PaymentMethod": [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ][i % 4],
        "MonthlyCharges": round(25 + (i % 95) * 0.9, 2),
        "TotalCharges": round((1 + (i % 72)) * (25 + (i % 95) * 0.9), 2),
    }


def _docker(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True)


def _docker_disponivel() -> bool:
    return _docker(["docker", "info"]).returncode == 0


@pytest.mark.e2e
def test_e2e_api_1000_chamadas_simples():
    total_calls = 1000

    if not _docker_disponivel():
        pytest.skip("Docker não está disponível neste ambiente.")

    # Limpa container antigo, se existir.
    _docker(["docker", "rm", "-f", CONTAINER_NAME])

    build = _docker(["docker", "build", "-t", IMAGE_TAG, "."])
    assert build.returncode == 0, f"Build falhou:\n{build.stdout}\n{build.stderr}"

    run = _docker(
        [
            "docker",
            "run",
            "-d",
            "--name",
            CONTAINER_NAME,
            "-p",
            "8010:8000",
            IMAGE_TAG,
        ]
    )
    assert run.returncode == 0, f"Run falhou:\n{run.stdout}\n{run.stderr}"

    try:
        # Aguarda a API ficar pronta no health.
        ready = False
        for _ in range(30):
            try:
                health = requests.get(f"{BASE_URL}/health", timeout=2)
                if health.status_code == 200:
                    ready = True
                    break
            except Exception:
                pass
            time.sleep(1)

        assert ready, "API não ficou pronta no /health após 30s"

        ok_200 = 0
        pred_true = 0
        pred_false = 0
        latencias_ms = []

        start_all = time.perf_counter()

        for i in range(total_calls):
            payload = _build_payload(i)

            start_call = time.perf_counter()
            resp = requests.post(f"{BASE_URL}/predict", json=payload, timeout=10)
            latencias_ms.append((time.perf_counter() - start_call) * 1000)

            if resp.status_code == 200:
                ok_200 += 1
                pred = resp.json().get("prediction")
                if pred is True:
                    pred_true += 1
                elif pred is False:
                    pred_false += 1

        total_time = time.perf_counter() - start_all
        media_ms = sum(latencias_ms) / len(latencias_ms)
        rps = total_calls / total_time

        print("\n===== RESULTADO E2E SIMPLES =====")
        print(f"total_calls={total_calls}")
        print(f"status_200={ok_200}")
        print(f"prediction_true={pred_true}")
        print(f"prediction_false={pred_false}")
        print(f"tempo_total_s={total_time:.2f}")
        print(f"rps={rps:.2f}")
        print(f"latencia_media_ms={media_ms:.2f}")

        assert ok_200 == total_calls, "Nem todas as chamadas retornaram 200"
        assert pred_true + pred_false == total_calls, "Nem todas as respostas tiveram prediction válido"

    finally:
        _docker(["docker", "rm", "-f", CONTAINER_NAME])
