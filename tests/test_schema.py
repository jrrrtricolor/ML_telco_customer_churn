import pandas as pd
import pytest

pa = pytest.importorskip("pandera.pandas", reason="pandera ainda não está instalado")


def test_schema_telco_minimo():
    dados = pd.DataFrame(
        {
            "gender": ["Male", "Female"],
            "SeniorCitizen": [0, 1],
            "tenure": [12, 24],
            "MonthlyCharges": [55.2, 72.4],
            "TotalCharges": [662.4, 1737.6],
            "Churn": [0, 1],
        }
    )

    # Schema enxuto para validar as colunas críticas do pipeline.
    schema = pa.DataFrameSchema(
        {
            "gender": pa.Column(str, checks=pa.Check.isin(["Male", "Female"])),
            "SeniorCitizen": pa.Column(int, checks=pa.Check.isin([0, 1])),
            "tenure": pa.Column(int, checks=pa.Check.ge(0)),
            "MonthlyCharges": pa.Column(float, checks=pa.Check.ge(0)),
            "TotalCharges": pa.Column(float, checks=pa.Check.ge(0)),
            "Churn": pa.Column(int, checks=pa.Check.isin([0, 1])),
        }
    )

    resultado = schema.validate(dados)

    assert len(resultado) == len(dados)
