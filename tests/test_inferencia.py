"""Testes da camada de inferência."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.models.treino import salvar_pacote_modelo, treinar_modelo_baseline
from src.servico.inferencia import carregar_pacote_modelo, prever_por_payload


def test_prever_por_payload_retorna_saida_esperada(tmp_path: Path) -> None:
    """Valida formato e faixa das predições retornadas pela inferência."""
    modelo, metricas = treinar_modelo_baseline(
        caminho_dados="data/processed/telco_churn_encoded.csv"
    )
    caminho_modelo = tmp_path / "modelo.joblib"
    salvar_pacote_modelo(modelo, metricas, caminho_modelo)
    pacote = carregar_pacote_modelo(caminho_modelo)

    dataframe = pd.read_csv("data/processed/telco_churn_encoded.csv").drop(
        columns=["Churn"]
    )
    carga = dataframe.head(5).to_dict(orient="records")
    resposta = prever_por_payload(carga, pacote)

    assert len(resposta) == 5
    for item in resposta:
        assert "predicao" in item
        assert "probabilidade" in item
        assert item["predicao"] in {0, 1}
        assert 0.0 <= item["probabilidade"] <= 1.0
