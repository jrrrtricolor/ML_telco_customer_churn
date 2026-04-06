import pandas as pd

from fastapi import FastAPI
from enum import Enum
from pydantic import BaseModel
import mlflow
import mlflow.sklearn

# Usa banco SQLite (mais estável)
TRACKING_URI = "sqlite:///mlflow.db"

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_registry_uri(TRACKING_URI)

model_name = "churn_mlp"
model_version = "latest"

# Load the model from the Model Registry
model_uri = f"models:/{model_name}/{model_version}"
model = mlflow.sklearn.load_model(model_uri)

app = FastAPI()

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

    def to_dict(self):
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
def read_health():
    return {"status": "ok"}

@app.post("/predict")
def read_predict(prediction_request: PredictionRequest) -> PredictionResponse:
    global model

    data_dict = prediction_request.to_dict()
    data = pd.DataFrame([list(data_dict.values())], columns=data_dict.keys())
    prediction = model.predict(data)
    return PredictionResponse(prediction=bool(prediction[0]))
