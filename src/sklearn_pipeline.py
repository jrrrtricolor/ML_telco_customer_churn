from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data_classifier import DataClassifier


def criar_pipeline_modelo(modelo):
    """
    Cria um pipeline sklearn simples e reprodutível.

    Fluxo:
    - imputação de valores numéricos ausentes
    - normalização das features
    - modelo de classificação
    """

    pipeline = Pipeline(
        [
            ("classifier", DataClassifier()),
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("modelo", modelo),
        ]
    )

    return pipeline
