"""Treino com múltiplos parâmetros para modelos baseline e árvores."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from src.data.ingestao import carregar_dados_brutos
from src.evaluation.metricas import avaliar_classificacao
from src.features.preprocessamento import (
    construir_preprocessador,
    preparar_atributos_alvo,
)


@dataclass(frozen=True)
class DefinicaoModelo:
    """Representa uma definição de modelo e seu grid de parâmetros."""

    nome: str
    classificador: Any
    grade_parametros: dict[str, list[Any]]


def obter_raiz_projeto() -> Path:
    """Retorna o caminho da raiz do repositório."""
    return Path(__file__).resolve().parents[2]


def obter_definicoes_modelos(estado_aleatorio: int) -> list[DefinicaoModelo]:
    """Retorna modelos candidatos e grades de parâmetros."""
    return [
        DefinicaoModelo(
            nome="dummy",
            classificador=DummyClassifier(),
            grade_parametros={"classificador__strategy": ["most_frequent", "prior"]},
        ),
        DefinicaoModelo(
            nome="regressao_logistica",
            classificador=LogisticRegression(
                random_state=estado_aleatorio,
                max_iter=1500,
                class_weight="balanced",
            ),
            grade_parametros={"classificador__C": [0.1, 1.0, 3.0]},
        ),
        DefinicaoModelo(
            nome="arvore_random_forest",
            classificador=RandomForestClassifier(
                random_state=estado_aleatorio,
                class_weight="balanced",
            ),
            grade_parametros={
                "classificador__n_estimators": [200, 400],
                "classificador__max_depth": [None, 8, 16],
                "classificador__min_samples_leaf": [1, 5],
            },
        ),
        DefinicaoModelo(
            nome="arvore_gradient_boosting",
            classificador=GradientBoostingClassifier(random_state=estado_aleatorio),
            grade_parametros={
                "classificador__n_estimators": [100, 200],
                "classificador__learning_rate": [0.05, 0.1],
                "classificador__max_depth": [2, 3],
            },
        ),
    ]


def treinar_e_comparar_modelos(
    caminho_dados: str | Path = "data/processed/telco_churn_encoded.csv",
    estado_aleatorio: int = 42,
    usar_grade_reduzida: bool = False,
    n_splits_cv: int = 5,
) -> dict[str, Any]:
    """Treina múltiplos modelos com busca de parâmetros e retorna comparação.

    A função devolve dois destaques:
    - melhor_modelo: maior ROC AUC (visão estatística global).
    - modelo_recomendado_retenção: melhor equilíbrio para negócio de churn.
    """
    dataframe = carregar_dados_brutos(caminho_dados)
    atributos, alvo = preparar_atributos_alvo(dataframe)

    x_treino, x_teste, y_treino, y_teste = train_test_split(
        atributos,
        alvo,
        test_size=0.2,
        random_state=estado_aleatorio,
        stratify=alvo,
    )
    preprocessador = construir_preprocessador(atributos)
    validacao_cruzada = StratifiedKFold(
        n_splits=n_splits_cv, shuffle=True, random_state=estado_aleatorio
    )

    resultados: dict[str, Any] = {"modelos": {}}
    melhor_nome = ""
    melhor_roc_auc = -1.0
    melhor_estimador = None

    definicoes_modelos = obter_definicoes_modelos(estado_aleatorio)
    if usar_grade_reduzida:
        definicoes_modelos = [
            DefinicaoModelo(
                nome=definicao.nome,
                classificador=definicao.classificador,
                grade_parametros={
                    parametro: [valores[0]]
                    for parametro, valores in definicao.grade_parametros.items()
                },
            )
            for definicao in definicoes_modelos
        ]

    for definicao in definicoes_modelos:
        pipeline = Pipeline(
            steps=[
                ("preprocessador", preprocessador),
                ("classificador", definicao.classificador),
            ]
        )

        busca = GridSearchCV(
            estimator=pipeline,
            param_grid=definicao.grade_parametros,
            scoring="roc_auc",
            cv=validacao_cruzada,
            n_jobs=-1,
            refit=True,
        )
        busca.fit(x_treino, y_treino)

        melhor_pipeline = busca.best_estimator_
        predicao = melhor_pipeline.predict(x_teste)
        if hasattr(melhor_pipeline, "predict_proba"):
            escore = melhor_pipeline.predict_proba(x_teste)[:, 1]
        else:
            escore = melhor_pipeline.decision_function(x_teste)

        metricas = avaliar_classificacao(y_teste, predicao, escore)
        resultados["modelos"][definicao.nome] = {
            "metricas_teste": metricas,
            "melhores_parametros": busca.best_params_,
            "melhor_score_cv_roc_auc": float(busca.best_score_),
        }

        if metricas["roc_auc"] > melhor_roc_auc:
            melhor_roc_auc = metricas["roc_auc"]
            melhor_nome = definicao.nome
            melhor_estimador = melhor_pipeline

    # Regra simples e explicável para retenção:
    # 1) maior recall (capturar mais churners),
    # 2) desempate por PR AUC,
    # 3) desempate por F1.
    itens_modelos = list(resultados["modelos"].items())
    itens_ordenados_retencao = sorted(
        itens_modelos,
        key=lambda item: (
            item[1]["metricas_teste"]["recall"],
            item[1]["metricas_teste"]["pr_auc"],
            item[1]["metricas_teste"]["f1"],
        ),
        reverse=True,
    )
    modelo_recomendado_retencao = itens_ordenados_retencao[0][0]

    resultados["melhor_modelo"] = melhor_nome
    resultados["melhor_roc_auc_teste"] = melhor_roc_auc
    resultados["modelo_recomendado_retencao"] = modelo_recomendado_retencao
    resultados["estimador_melhor_modelo"] = melhor_estimador
    return resultados


def salvar_resultados_comparacao(
    resultados: dict[str, Any],
    caminho_saida: str | Path | None = None,
) -> Path:
    """Salva artefato com comparação e melhor modelo treinado."""
    caminho_destino = (
        Path(caminho_saida)
        if caminho_saida is not None
        else obter_raiz_projeto()
        / "models"
        / "trained_models"
        / "comparacao_modelos.joblib"
    )
    caminho_destino.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(resultados, caminho_destino)
    return caminho_destino


def main() -> None:
    """Executa treino com múltiplos parâmetros e persiste resultados."""
    resultados = treinar_e_comparar_modelos()
    salvar_resultados_comparacao(resultados)
    print(f"Melhor ROC AUC: {resultados['melhor_modelo']}")
    print(
        f"Modelo recomendado para retenção: {resultados['modelo_recomendado_retencao']}"
    )


if __name__ == "__main__":
    main()
