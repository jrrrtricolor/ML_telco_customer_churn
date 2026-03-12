"""Testes unitários para preprocessamento de dados."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.preprocessamento import (
    construir_preprocessador,
    normalizar_alvo,
    preparar_atributos_alvo,
)


def test_normalizar_alvo_mapeia_yes_no() -> None:
    """Valida mapeamento de rótulos de churn para 0 e 1."""
    alvo = pd.Series(["Yes", "No", "Yes", "No"])
    alvo_normalizado = normalizar_alvo(alvo)
    assert alvo_normalizado.tolist() == [1, 0, 1, 0]


def test_normalizar_alvo_valor_invalido() -> None:
    """Garante erro quando o alvo possui valor fora do domínio esperado."""
    alvo = pd.Series(["Yes", "Talvez"])
    with pytest.raises(ValueError):
        normalizar_alvo(alvo)


def test_preparar_atributos_alvo_remove_customerid() -> None:
    """Confere separação correta entre atributos e alvo."""
    dataframe = pd.DataFrame(
        {
            "customerID": ["A", "B"],
            "MonthlyCharges": [70.0, 20.0],
            "Contract": ["Month-to-month", "Two year"],
            "Churn": ["Yes", "No"],
        }
    )
    atributos, alvo = preparar_atributos_alvo(dataframe)
    assert "customerID" not in atributos.columns
    assert "Churn" not in atributos.columns
    assert alvo.tolist() == [1, 0]


def test_construir_preprocessador_com_imputacao() -> None:
    """Valida pipeline com imputação para dados numéricos e categóricos."""
    atributos = pd.DataFrame(
        {
            "tenure": [1.0, np.nan, 4.0],
            "MonthlyCharges": [40.0, 55.0, np.nan],
            "Contract": ["Month-to-month", None, "Two year"],
        }
    )
    preprocessador = construir_preprocessador(atributos)
    matriz_transformada = preprocessador.fit_transform(atributos)
    assert matriz_transformada.shape[0] == 3
    assert not np.isnan(matriz_transformada).any()
