"""Utilitários para carregamento de dados do projeto."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def obter_raiz_projeto() -> Path:
    """Retorna a raiz do repositório com base na localização deste módulo."""
    return Path(__file__).resolve().parents[2]


def caminho_padrao_dados_brutos() -> Path:
    """Retorna o caminho padrão do dataset bruto de churn."""
    return obter_raiz_projeto() / "data" / "raw" / "Telco_Customer_Churn.csv"


def carregar_dados_brutos(caminho: str | Path | None = None) -> pd.DataFrame:
    """Carrega dados brutos e aplica normalização mínima de tipos."""
    caminho_dados = (
        Path(caminho) if caminho is not None else caminho_padrao_dados_brutos()
    )
    dataframe = pd.read_csv(caminho_dados)

    if "TotalCharges" in dataframe.columns:
        dataframe["TotalCharges"] = pd.to_numeric(
            dataframe["TotalCharges"], errors="coerce"
        )

    return dataframe
