import logging
import os
import time
from enum import Enum
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, Request
from pydantic import BaseModel, Field

from src.config.logging_config import setup_logging

setup_logging()

logger = logging.getLogger(__name__)

# Usa banco SQLite (mais estável)
TRACKING_URI = "sqlite:///mlflow.db"
MODEL_NAME = "churn_mlp"
MODEL_VERSION = "latest"

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_registry_uri(TRACKING_URI)

_MODEL = None


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
        logger.info("Carregando modelo a partir de MODEL_URI=%s", model_uri)
        return mlflow.sklearn.load_model(model_uri)

    registry_uri = f"models:/{MODEL_NAME}/{MODEL_VERSION}"
    try:
        logger.info("Carregando modelo registrado em %s", registry_uri)
        return mlflow.sklearn.load_model(registry_uri)
    except Exception:
        local_uri = _resolve_local_model_uri()
        if local_uri is None:
            raise
        logger.info("Carregando modelo local em %s", local_uri)
        return mlflow.sklearn.load_model(local_uri)


def _get_model() -> object:
    global _MODEL

    # Carrega sob demanda para o /health funcionar mesmo antes do modelo.
    if _MODEL is None:
        _MODEL = _load_model()

    return _MODEL


app = FastAPI()


@app.middleware("http")
async def log_latency(request: Request, call_next):
    started_at = time.perf_counter()
    response = await call_next(request)
    latency_ms = (time.perf_counter() - started_at) * 1000

    logger.info(
        "request method=%s path=%s status_code=%s latency_ms=%.2f",
        request.method,
        request.url.path,
        response.status_code,
        latency_ms,
    )

    return response


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
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: YesNoEnum
    Dependents: YesNoEnum
    tenure: int = Field(ge=0)
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
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)

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
    model = _get_model()
    prediction = model.predict(data)
    return PredictionResponse(prediction=bool(prediction[0]))
