import logging
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, plot_tree

from src.models.modelos import Modelos

LOGGER = logging.getLogger(__name__)


class Treino:

    def __init__(
        self,
        variaveis_explicaveis: pd.DataFrame,
        variavel_target: pd.Series,
        random_state: int = 42
    ):

        self.X = variaveis_explicaveis
        self.y = variavel_target
        self.random_state = random_state

        self.x_treino = None
        self.x_teste = None
        self.y_treino = None
        self.y_teste = None

    # --------------------------------------------------
    # Split dos dados
    # --------------------------------------------------

    def split_dados(self, test_size: float = 0.2) -> None:

        self.x_treino, self.x_teste,
        self.y_treino, self.y_teste = train_test_split(
            self.X,
            self.y,
            test_size=test_size,
            stratify=self.y,
            random_state=self.random_state
        )

        LOGGER.info("Dados divididos em treino e teste")

    # --------------------------------------------------
    # Criar modelos
    # --------------------------------------------------

    def criar_modelos(self) -> dict[str, Pipeline]:

        modelos_base = {

            "dummy": DummyClassifier(strategy="most_frequent"),

            "decision_tree": DecisionTreeClassifier(
                random_state=self.random_state
            ),

            "random_forest": RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state
            ),

            "knn": KNeighborsClassifier(
                n_neighbors=5
            )
        }

        modelos_treinados: dict[str, Pipeline] = {}

        for nome, modelo in modelos_base.items():
            pipeline_modelo = Modelos.criar_pipeline(
                modelo=modelo,
                variaveis_explicaveis=self.X,
            )

            pipeline_modelo.fit(self.x_treino, self.y_treino)

            modelos_treinados[nome] = pipeline_modelo

            LOGGER.info("Modelo treinado: %s", nome)

        return modelos_treinados

    @staticmethod
    def obter_scores(modelo: Pipeline,
                     x_teste: pd.DataFrame) -> np.ndarray | None:
        if hasattr(modelo, "predict_proba"):
            probabilidades = modelo.predict_proba(x_teste)
            if probabilidades.ndim == 2 and probabilidades.shape[1] > 1:
                return probabilidades[:, 1]

            return np.ravel(probabilidades)

        if hasattr(modelo, "decision_function"):
            return np.ravel(modelo.decision_function(x_teste))

        return None

    # --------------------------------------------------
    # Avaliar modelos
    # --------------------------------------------------

    def avaliar_modelos(self, modelos: dict[str, Pipeline]) -> pd.DataFrame:

        resultados = []

        for nome, modelo in modelos.items():

            y_pred = modelo.predict(self.x_teste)
            y_score = self.obter_scores(modelo, self.x_teste)

            metricas = {
                "modelo": nome,
                "accuracy": accuracy_score(self.y_teste, y_pred),
                "precision": precision_score(self.y_teste,
                                             y_pred, zero_division=0),
                "recall": recall_score(self.y_teste, y_pred, zero_division=0),
                "f1": f1_score(self.y_teste, y_pred, zero_division=0),
                "roc_auc": np.nan,
                "pr_auc": np.nan,
            }

            if y_score is not None:
                try:
                    metricas["roc_auc"] = roc_auc_score(self.y_teste, y_score)
                except ValueError:
                    LOGGER.warning(
                        "Nao foi possivel calcular ROC-AUC para o modelo %s",
                        nome)

                try:
                    metricas["pr_auc"] = average_precision_score(
                        self.y_teste, y_score)
                except ValueError:
                    LOGGER.warning(
                        "Nao foi possivel calcular PR-AUC para o modelo %s",
                        nome)

            resultados.append(metricas)

        df_resultados = pd.DataFrame(resultados)

        df_resultados = df_resultados.sort_values(
            by=["roc_auc", "pr_auc", "f1", "recall", "accuracy"],
            ascending=False,
            na_position="last",
        )

        df_resultados = df_resultados.reset_index(drop=True)

        return df_resultados

    # --------------------------------------------------
    # Plotar árvore de decisão
    # --------------------------------------------------

    def plotar_arvore_decisao(self, modelo: Pipeline) -> None:

        if not isinstance(modelo, Pipeline):
            raise TypeError(
                "plotar_arvore_decisao espera um modelo sklearn Pipeline.")

        arvore = modelo.named_steps.get("modelo")
        preprocessador = modelo.named_steps.get("preprocessador")

        if not isinstance(arvore, DecisionTreeClassifier):
            raise TypeError(
                "O estimador final nao e uma DecisionTreeClassifier.")

        feature_names = self.X.columns.tolist()
        if preprocessador is not None and hasattr(preprocessador,
                                                  "get_feature_names_out"):
            feature_names = preprocessador.get_feature_names_out().tolist()

        class_names = [str(classe) for classe in np.unique(self.y)]

        plt.figure(figsize=(20, 10))

        plot_tree(
            arvore,
            filled=True,
            feature_names=feature_names,
            class_names=class_names,
        )

        plt.title("Árvore de Decisão")

        plt.show()

    # --------------------------------------------------
    # Salvar modelo
    # --------------------------------------------------

    @staticmethod
    def salvar_modelo(modelo, nome_modelo: str,
                      path: str = "models/trained_models") -> None:

        os.makedirs(path, exist_ok=True)

        file_path = os.path.join(path, f"{nome_modelo}.joblib")

        joblib.dump(modelo, file_path)

        LOGGER.info("Modelo salvo em: %s", file_path)
