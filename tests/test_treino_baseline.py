"""Testes de treino baseline e persistência de modelo."""

from __future__ import annotations

from pathlib import Path

import joblib

from src.models.treino import salvar_pacote_modelo, treinar_modelo_baseline


def test_treinar_modelo_baseline_retorna_metricas() -> None:
    """Valida retorno de métricas em faixa esperada."""
    _, metricas = treinar_modelo_baseline(
        caminho_dados="data/processed/telco_churn_encoded.csv"
    )
    chaves_esperadas = {"roc_auc", "pr_auc", "f1", "accuracy", "precision", "recall"}
    assert chaves_esperadas.issubset(metricas.keys())
    for valor in metricas.values():
        assert 0.0 <= valor <= 1.0


def test_salvar_pacote_modelo_persiste_artefato(tmp_path: Path) -> None:
    """Valida serialização do pacote com modelo e métricas."""
    modelo, metricas = treinar_modelo_baseline(
        caminho_dados="data/processed/telco_churn_encoded.csv"
    )
    caminho_saida = tmp_path / "pacote_modelo.joblib"
    salvar_pacote_modelo(modelo, metricas, caminho_saida)

    assert caminho_saida.exists()
    pacote = joblib.load(caminho_saida)
    assert "modelo" in pacote
    assert "metricas" in pacote
