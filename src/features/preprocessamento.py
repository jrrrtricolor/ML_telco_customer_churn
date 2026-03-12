"""Funções de preparação de atributos e alvo."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

COLUNA_ALVO = "Churn"


def normalizar_alvo(alvo: pd.Series) -> pd.Series:
    """Converte rótulos de churn para representação numérica (0/1)."""
    mapeamento = {"Yes": 1, "No": 0, 1: 1, 0: 0}
    alvo_normalizado = alvo.map(mapeamento)
    if alvo_normalizado.isna().any():
        raise ValueError("A coluna alvo contém valores não suportados para churn.")
    return alvo_normalizado.astype(int)


def preparar_atributos_alvo(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separa o dataframe em atributos e alvo."""
    if COLUNA_ALVO not in dataframe.columns:
        raise ValueError(f"Coluna alvo obrigatória ausente: {COLUNA_ALVO}")

    dataframe_limpo = dataframe.copy()
    dataframe_limpo = dataframe_limpo.drop(columns=["customerID"], errors="ignore")
    dataframe_limpo = dataframe_limpo.dropna(subset=[COLUNA_ALVO])

    alvo = normalizar_alvo(dataframe_limpo[COLUNA_ALVO])
    atributos = dataframe_limpo.drop(columns=[COLUNA_ALVO])

    return atributos, alvo


def construir_preprocessador(atributos: pd.DataFrame) -> ColumnTransformer:
    """Cria preprocessador sklearn para colunas numéricas e categóricas."""
    colunas_numericas = atributos.select_dtypes(include=["number"]).columns.tolist()
    colunas_categoricas = [
        coluna for coluna in atributos.columns if coluna not in colunas_numericas
    ]

    transformadores: list[tuple[str, Any, list[str]]] = []
    if colunas_numericas:
        pipeline_numerico = Pipeline(
            steps=[
                ("imputador", SimpleImputer(strategy="median")),
                ("escalador", StandardScaler()),
            ]
        )
        transformadores.append(("numerico", pipeline_numerico, colunas_numericas))
    if colunas_categoricas:
        pipeline_categorico = Pipeline(
            steps=[
                ("imputador", SimpleImputer(strategy="most_frequent")),
                (
                    "codificador",
                    OneHotEncoder(handle_unknown="ignore"),
                ),
            ]
        )
        transformadores.append(
            (
                "categorico",
                pipeline_categorico,
                colunas_categoricas,
            )
        )

    return ColumnTransformer(transformers=transformadores, remainder="drop")
