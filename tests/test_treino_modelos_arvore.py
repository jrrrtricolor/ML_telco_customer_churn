"""Testes para comparação de modelos com busca de parâmetros."""

from __future__ import annotations

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

import src.models.treino_modelos_arvore as modulo_treino
from src.models.treino_modelos_arvore import DefinicaoModelo, treinar_e_comparar_modelos


def test_treinar_e_comparar_modelos_retorna_estrutura(monkeypatch) -> None:
    """Executa comparação com grade reduzida para validar estrutura de retorno."""

    def definicoes_reduzidas(estado_aleatorio: int) -> list[DefinicaoModelo]:
        return [
            DefinicaoModelo(
                nome="dummy",
                classificador=DummyClassifier(),
                grade_parametros={"classificador__strategy": ["most_frequent"]},
            ),
            DefinicaoModelo(
                nome="regressao_logistica",
                classificador=LogisticRegression(
                    random_state=estado_aleatorio,
                    max_iter=500,
                    class_weight="balanced",
                ),
                grade_parametros={"classificador__C": [1.0]},
            ),
        ]

    monkeypatch.setattr(modulo_treino, "obter_definicoes_modelos", definicoes_reduzidas)
    resultados = treinar_e_comparar_modelos(
        caminho_dados="data/processed/telco_churn_encoded.csv"
    )

    assert "modelos" in resultados
    assert "melhor_modelo" in resultados
    assert resultados["melhor_modelo"] in {"dummy", "regressao_logistica"}
    assert "regressao_logistica" in resultados["modelos"]
