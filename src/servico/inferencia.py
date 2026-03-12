"""Funções de inferência para a camada de serviço."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd


def carregar_pacote_modelo(caminho: str | Path) -> dict[str, Any]:
    """Carrega do disco um pacote de modelo persistido."""
    pacote_modelo = joblib.load(caminho)
    if "modelo" not in pacote_modelo:
        raise ValueError("Pacote de modelo inválido: chave 'modelo' ausente.")
    return pacote_modelo


def prever_por_payload(
    carga: list[dict[str, Any]],
    pacote_modelo: dict[str, Any],
) -> list[dict[str, float | int]]:
    """Gera predições de churn a partir de um payload em formato JSON."""
    modelo = pacote_modelo["modelo"]
    atributos = pd.DataFrame(carga)

    probabilidades = modelo.predict_proba(atributos)[:, 1]
    predicoes = (probabilidades >= 0.5).astype(int)

    return [
        {"predicao": int(predicao), "probabilidade": float(probabilidade)}
        for predicao, probabilidade in zip(predicoes, probabilidades)
    ]
