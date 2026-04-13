import logging

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


class Avaliador:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def calcular_metricas(self, y_true, y_pred, y_prob):
        """
        Calcula métricas a partir das previsões já geradas.

        Esse método permite compatibilidade com modelos sklearn e PyTorch,
        pois não depende diretamente do modelo.
        """
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        roc_auc = roc_auc_score(y_true, y_prob)
        pr_auc = average_precision_score(y_true, y_prob)

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
        }

    def calcular_custo_negocio(self, y_true, y_pred):
        """
        Calcula o impacto financeiro das previsões do modelo.

        Dicionário:
        * FP = Falso Positivo
        * FN = Falso Negativo
        * 100 = custo de retenção por cliente
        * 840 = perda média por cliente que cancela

        Regras:
        - FP (Falso Positivo): cliente não ia sair, mas recebeu oferta,
          gerando custo menor (R$ 100)
        - FN (Falso Negativo): cliente ia sair e não foi identificado,
          gerando custo alto (R$ 840)

        Objetivo:
        Minimizar esse custo, principalmente evitando perder clientes (FN).
        """

        # Clientes que não iam sair, mas o modelo disse que iam.
        falsos_positivos = ((y_pred == 1) & (y_true == 0)).sum()

        # Clientes que iam sair, mas o modelo não identificou.
        falsos_negativos = ((y_pred == 0) & (y_true == 1)).sum()

        # Cálculo do custo total.
        custo_total = (falsos_positivos * 100) + (falsos_negativos * 840)

        return custo_total

    def avaliar(self, modelo, x_teste, y_teste):
        y_pred = modelo.predict(x_teste)

        # Alguns modelos têm predict_proba.
        if hasattr(modelo, "predict_proba"):
            y_prob = modelo.predict_proba(x_teste)[:, 1]
        else:
            y_prob = None

        return self.calcular_metricas(y_teste, y_pred, y_prob)
