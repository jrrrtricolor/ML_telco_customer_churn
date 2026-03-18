import logging
import os
from hashlib import sha256
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
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
        random_state: int = 42,
        valor_medio_churn_evitado: float = 1000.0,
        custo_contato_retencao: float = 40.0,
        taxa_sucesso_retencao: float = 0.35,
    ):

        self.X = variaveis_explicaveis
        self.y = variavel_target
        self.random_state = random_state
        self.valor_medio_churn_evitado = valor_medio_churn_evitado
        self.custo_contato_retencao = custo_contato_retencao
        self.taxa_sucesso_retencao = taxa_sucesso_retencao
        self.test_size: float | None = None

        self.x_treino = None
        self.x_teste = None
        self.y_treino = None
        self.y_teste = None

    # --------------------------------------------------
    # Split dos dados
    # --------------------------------------------------

    def split_dados(self, test_size: float = 0.2) -> None:

        self.test_size = test_size

        (self.x_treino,
         self.x_teste,
         self.y_treino,
         self.y_teste) = train_test_split(
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

    def criar_modelos(self, max_depth: int | None, kn_neighbors: int | None) -> dict[str, Pipeline]:

        modelos_base = {

            "dummy": DummyClassifier(strategy="most_frequent"),

            "decision_tree": DecisionTreeClassifier(
                random_state=self.random_state
                ,max_depth=max_depth
            ),

            "random_forest": RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state,
                max_depth=max_depth,

            ),

            "knn": KNeighborsClassifier(
                n_neighbors=kn_neighbors
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
    def calcular_hash_arquivo(caminho_arquivo: str) -> str:

        arquivo_hash = sha256()

        with Path(caminho_arquivo).open("rb") as arquivo:
            for bloco in iter(lambda: arquivo.read(8192), b""):
                arquivo_hash.update(bloco)

        return arquivo_hash.hexdigest()

    @staticmethod
    def extrair_parametros_estimador(
        modelo: Pipeline,
    ) -> dict[str, str | int | float | bool]:

        estimador = modelo.named_steps.get("modelo")
        if estimador is None:
            return {}

        parametros = estimador.get_params()
        parametros_filtrados: dict[str, str | int | float | bool] = {}

        for chave, valor in parametros.items():
            if isinstance(valor, (str, int, float, bool)):
                parametros_filtrados[chave] = valor

        return parametros_filtrados

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
            by=[
                "roc_auc",
                "pr_auc",
                "f1",
                "recall",
                "accuracy"
            ],
            ascending=False,
            na_position="last",
        )

        df_resultados = df_resultados.reset_index(drop=True)

        return df_resultados

    def registrar_experimentos_mlflow(
        self,
        modelos: dict[str, Pipeline],
        resultados: pd.DataFrame,
        dataset_path: str,
        nome_experimento: str = "telco_churn_fase1",
        tracking_uri: str | None = None,
    ) -> None:

        if self.test_size is None:
            raise ValueError(
                "Execute split_dados antes de "
                "registrar experimentos no MLflow.")

        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        LOGGER.info(
            "MLflow tracking URI em uso: %s",
            mlflow.get_tracking_uri(),
        )

        mlflow.set_experiment(nome_experimento)

        dataset_hash = self.calcular_hash_arquivo(dataset_path)
        dataset_info = {
            "dataset_path": dataset_path,
            "dataset_sha256": dataset_hash,
            "linhas": int(self.X.shape[0]),
            "colunas": int(self.X.shape[1]),
            "target_distribuicao": {
                "classe_0": int((self.y == 0).sum()),
                "classe_1": int((self.y == 1).sum()),
            },
        }

        for _, linha_resultado in resultados.iterrows():
            nome_modelo = str(linha_resultado["modelo"])
            modelo = modelos[nome_modelo]

            with mlflow.start_run(run_name=f"{nome_modelo}"):
                mlflow.log_param("modelo", nome_modelo)
                mlflow.log_param("random_state", self.random_state)
                mlflow.log_param("test_size", self.test_size)
                mlflow.log_param("dataset_path", dataset_path)
                mlflow.log_param("dataset_sha256", dataset_hash)

                parametros_estimador = self.extrair_parametros_estimador(
                    modelo)
                if parametros_estimador:
                    mlflow.log_params(parametros_estimador)

                metricas_log: dict[str, float] = {}
                # for coluna in resultados.columns:
                #     valor = linha_resultado[coluna]
                #     if coluna == "modelo":
                #         continue
                #
                #     if pd.notna(valor):
                #         metricas_log[coluna] = float(valor)

                if metricas_log:
                    mlflow.log_metrics(metricas_log)

                mlflow.log_dict(dataset_info, "dataset_version.json")
                mlflow.sklearn.log_model(modelo, name="modelo")

                LOGGER.info(
                    "Run MLflow registrado para o modelo: %s", nome_modelo)

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
