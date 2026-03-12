"""Ponto de entrada para treinamento de baseline."""

from __future__ import annotations

from pathlib import Path

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.data.ingestao import carregar_dados_brutos
from src.evaluation.metricas import avaliar_classificacao
from src.features.preprocessamento import (
    construir_preprocessador,
    preparar_atributos_alvo,
)


def obter_raiz_projeto() -> Path:
    """Retorna o caminho da raiz do repositório."""
    return Path(__file__).resolve().parents[2]


def treinar_modelo_baseline(
    caminho_dados: str | Path | None = None,
    estado_aleatorio: int = 42,
) -> tuple[Pipeline, dict[str, float]]:
    """Treina baseline de regressão logística e retorna métricas.

    Este baseline é o "ponto de partida" para comparar modelos mais avançados.
    """
    dataframe = carregar_dados_brutos(caminho_dados)
    atributos, alvo = preparar_atributos_alvo(dataframe)

    # Split estratificado mantém a proporção de churn em treino e teste.
    x_treino, x_teste, y_treino, y_teste = train_test_split(
        atributos,
        alvo,
        test_size=0.2,
        random_state=estado_aleatorio,
        stratify=alvo,
    )

    # Pipeline mantém preprocessamento e modelo no mesmo fluxo.
    pipeline_modelo = Pipeline(
        steps=[
            ("preprocessador", construir_preprocessador(atributos)),
            (
                "classificador",
                LogisticRegression(
                    random_state=estado_aleatorio,
                    max_iter=1000,
                    class_weight="balanced",
                ),
            ),
        ]
    )
    pipeline_modelo.fit(x_treino, y_treino)

    predicao = pipeline_modelo.predict(x_teste)
    escore = pipeline_modelo.predict_proba(x_teste)[:, 1]
    metricas = avaliar_classificacao(y_teste, predicao, escore)

    return pipeline_modelo, metricas


def salvar_pacote_modelo(
    modelo: Pipeline,
    metricas: dict[str, float],
    caminho_saida: str | Path | None = None,
) -> Path:
    """Persiste modelo e métricas usando joblib."""
    caminho_destino = (
        Path(caminho_saida)
        if caminho_saida is not None
        else obter_raiz_projeto()
        / "models"
        / "trained_models"
        / "baseline_logistic.joblib"
    )
    caminho_destino.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"modelo": modelo, "metricas": metricas}, caminho_destino)
    return caminho_destino


def main() -> None:
    """Treina e persiste o modelo baseline."""
    modelo, metricas = treinar_modelo_baseline()
    salvar_pacote_modelo(modelo, metricas)


if __name__ == "__main__":
    main()
