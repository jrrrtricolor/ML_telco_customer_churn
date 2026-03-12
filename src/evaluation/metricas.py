"""Utilitários de avaliação de modelos."""

from __future__ import annotations

from typing import Any

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def avaliar_classificacao(
    alvo_real: list[int] | Any,
    predicao: list[int] | Any,
    escore: list[float] | Any,
) -> dict[str, float]:
    """Retorna métricas padrão de classificação para churn."""
    return {
        "roc_auc": float(roc_auc_score(alvo_real, escore)),
        "pr_auc": float(average_precision_score(alvo_real, escore)),
        "f1": float(f1_score(alvo_real, predicao)),
        "accuracy": float(accuracy_score(alvo_real, predicao)),
        "precision": float(precision_score(alvo_real, predicao, zero_division=0)),
        "recall": float(recall_score(alvo_real, predicao, zero_division=0)),
    }
