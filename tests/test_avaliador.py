import pandas as pd

from src.avaliador import Avaliador


def test_calcular_metricas_retorna_campos_obrigatorios():
    y_true = pd.Series([0, 1, 1, 0, 1])
    y_pred = [0, 1, 0, 0, 1]
    y_prob = [0.10, 0.90, 0.40, 0.20, 0.80]

    avaliador = Avaliador()
    metricas = avaliador.calcular_metricas(y_true, y_pred, y_prob)

    assert set(metricas.keys()) == {
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
    }


def test_calcular_custo_negocio_computa_fp_fn():
    y_true = pd.Series([0, 1, 1, 0])
    y_pred = pd.Series([1, 0, 1, 0])

    avaliador = Avaliador()
    custo = avaliador.calcular_custo_negocio(y_true, y_pred)

    # 1 FP (100) + 1 FN (840) = 940
    assert custo == 940

