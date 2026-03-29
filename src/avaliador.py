import logging
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)


class Avaliador:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def _calcular_metricas(self, y_true, y_pred, y_prob):
        resultado = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred),
        }

        if y_prob is not None:
            # Cálculo do AUC-ROC, se as probabilidades estiverem disponíveis
            resultado["AUC-ROC"] = roc_auc_score(y_true, y_prob)

            # Cálculo do AUC-PR, se as probabilidades estiverem disponíveis
            resultado["PR-AUC"] = average_precision_score(y_true, y_prob)

        self.logger.info("Resultado das métricas: %s", resultado)

        return resultado

    def calcular_custo_negocio(self, y_true, y_pred):
        """
            Calcula o impacto financeiro das previsões do modelo.
            Dicionário:
            * FP = Falso Positivo
            * FN = Falso Negativo
            * 100 = custo de retenção por cliente
            * 840 = perda média por cliente que cancela
            Regras:
            - FP (Falso Positivo): cliente não ia sair, mas recebeu oferta → custo menor (R$ 100) -- desconto
            - FN (Falso Negativo): cliente ia sair e não foi identificado → custo alto (R$ 840) -- Cliente perdido

            Objetivo:
            Minimizar esse custo, principalmente evitando perder clientes (FN).
            """

        # Clientes que não iam sair, mas o modelo disse que iam
        falsos_positivos = ((y_pred == 1) & (y_true == 0)).sum()

        # Clientes que iam sair, mas o modelo não identificou
        falsos_negativos = ((y_pred == 0) & (y_true == 1)).sum()

        # Cálculo do custo total
        custo_total = (falsos_positivos * 100) + (falsos_negativos * 840)

        return custo_total

    def avaliar(self, modelo,x_teste, y_teste):
        y_pred = modelo.predict(x_teste)

        # alguns modelos têm predict_proba
        if hasattr(modelo, "predict_proba"):
            y_prob = modelo.predict_proba(x_teste)[:, 1]
        else:
            y_prob = None

        return self._calcular_metricas(y_teste, y_pred, y_prob)

