import os
import time
from enum import Enum
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel

from src.config.api_logging_middleware import LoggingMiddleware
from src.config.logging_config import setup_api_logger
from src.metrics import AVG_CONFIDENCE, PREDICTION_DURATION, PREDICTIONS_TOTAL

LOGGER = setup_api_logger()

# Usa banco SQLite (mais estável)
TRACKING_URI = "sqlite:///mlflow.db"
MODEL_NAME = "churn_mlp"
MODEL_VERSION = "latest"

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_registry_uri(TRACKING_URI)


def _resolve_local_model_uri() -> str | None:
    # Fallback simples: usa o artefato mais recente salvo localmente em mlruns.
    candidates = sorted(
        Path("mlruns").glob("*/models/m-*/artifacts/model.pkl"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        return None
    return str(candidates[0].parent)


def _load_model() -> object:
    model_uri = os.getenv("MODEL_URI")
    if model_uri:
        return mlflow.sklearn.load_model(model_uri)

    registry_uri = f"models:/{MODEL_NAME}/{MODEL_VERSION}"
    try:
        return mlflow.sklearn.load_model(registry_uri)
    except Exception:
        local_uri = _resolve_local_model_uri()
        if local_uri is None:
            raise
        return mlflow.sklearn.load_model(local_uri)


model = _load_model()

app = FastAPI()
app.add_middleware(LoggingMiddleware)
Instrumentator().instrument(app).expose(app, endpoint="/metrics")


class GenderEnum(str, Enum):
    female = "Female"
    male = "Male"


class YesNoEnum(str, Enum):
    yes = "Yes"
    no = "No"


class PhoneServiceEnum(str, Enum):
    yes = "Yes"
    no = "No"


class MultipleLinesEnum(str, Enum):
    no = "No"
    yes = "Yes"
    no_phone_service = "No phone service"


class InternetServiceEnum(str, Enum):
    dsl = "DSL"
    fiber_optic = "Fiber optic"
    no = "No"


class InternetFeatureEnum(str, Enum):
    yes = "Yes"
    no = "No"
    no_internet_service = "No internet service"


class ContractEnum(str, Enum):
    month_to_month = "Month-to-month"
    one_year = "One year"
    two_year = "Two year"


class PaymentMethodEnum(str, Enum):
    electronic_check = "Electronic check"
    mailed_check = "Mailed check"
    bank_transfer_automatic = "Bank transfer (automatic)"
    credit_card_automatic = "Credit card (automatic)"


class PredictionRequest(BaseModel):
    gender: GenderEnum
    SeniorCitizen: int
    Partner: YesNoEnum
    Dependents: YesNoEnum
    tenure: int
    PhoneService: PhoneServiceEnum
    MultipleLines: MultipleLinesEnum
    InternetService: InternetServiceEnum
    OnlineSecurity: InternetFeatureEnum
    OnlineBackup: InternetFeatureEnum
    DeviceProtection: InternetFeatureEnum
    TechSupport: InternetFeatureEnum
    StreamingTV: InternetFeatureEnum
    StreamingMovies: InternetFeatureEnum
    Contract: ContractEnum
    PaperlessBilling: YesNoEnum
    PaymentMethod: PaymentMethodEnum
    MonthlyCharges: float
    TotalCharges: float

    def to_dict(self) -> dict[str, object]:
        return {
            "gender": self.gender.value,
            "SeniorCitizen": self.SeniorCitizen,
            "Partner": self.Partner.value,
            "Dependents": self.Dependents.value,
            "tenure": self.tenure,
            "PhoneService": self.PhoneService.value,
            "MultipleLines": self.MultipleLines.value,
            "InternetService": self.InternetService.value,
            "OnlineSecurity": self.OnlineSecurity.value,
            "OnlineBackup": self.OnlineBackup.value,
            "DeviceProtection": self.DeviceProtection.value,
            "TechSupport": self.TechSupport.value,
            "StreamingTV": self.StreamingTV.value,
            "StreamingMovies": self.StreamingMovies.value,
            "Contract": self.Contract.value,
            "PaperlessBilling": self.PaperlessBilling.value,
            "PaymentMethod": self.PaymentMethod.value,
            "MonthlyCharges": self.MonthlyCharges,
            "TotalCharges": self.TotalCharges,
        }


class PredictionResponse(BaseModel):
    prediction: bool


@app.get("/health")
def read_health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
def read_predict(prediction_request: PredictionRequest) -> PredictionResponse:
    data_dict = prediction_request.to_dict()
    data = pd.DataFrame([list(data_dict.values())], columns=list(data_dict.keys()))

    start_time = time.perf_counter()

    prediction = model.predict(data)
    prediction_proba = model.predict_proba(data)[:, 1]

    duration = time.perf_counter() - start_time

    LOGGER.info(
        "prediction_made",
        extra={
            "duration": duration,
            "confidence": prediction_proba[0],
            "input_data": data_dict,
            "prediction": bool(prediction[0]),
        },
    )

    PREDICTION_DURATION.observe(duration)
    PREDICTIONS_TOTAL.inc()
    AVG_CONFIDENCE.observe(prediction_proba[0])

    return PredictionResponse(prediction=bool(prediction[0]))
