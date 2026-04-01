from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


def criar_pipeline_modelo(modelo):
    """
    Cria um pipeline sklearn simples e reprodutível.

    Fluxo:
    - imputação de valores numéricos ausentes
    - normalização das features
    - modelo de classificação
    """

    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("modelo", modelo),
    ])