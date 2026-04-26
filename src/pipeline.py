import logging
import math

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate

from src.avaliador import Avaliador
from src.data_prep import DataPreprocessor

# Bibliotecas internas
from src.load import DataLoader
from src.model_factory import ModelFactory
from src.trainer import Trainer

# Usa banco SQLite (mais estável)
TRACKING_URI = "sqlite:///mlflow.db"

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_registry_uri(TRACKING_URI)


class Pipeline:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

        self.data_loader = DataLoader(data_path="data/raw/Telco_Customer_Churn.csv")

        self.data_prep = DataPreprocessor(
            pd_dataframe=self.data_loader.data,
            seed=2711,
            test_size=0.2,
        )

        self.avaliador = Avaliador()
        self.treinador = Trainer()
        self.model_factory = ModelFactory(seed=2711)

        self.cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=2711)

    @staticmethod
    def _valor_numerico_valido(valor: object) -> bool:
        return isinstance(valor, (int, float)) and math.isfinite(float(valor))

    def _calcular_metricas_cv(self, modelo, x_treino, y_treino) -> dict[str, float]:
        scoring = {
            "cv_roc_auc": "roc_auc",
            "cv_pr_auc": "average_precision",
            "cv_f1": "f1",
            "cv_precision": "precision",
            "cv_recall": "recall",
        }

        resultados_cv = cross_validate(
            estimator=modelo,
            X=x_treino,
            y=y_treino,
            cv=self.cv,
            scoring=scoring,
            n_jobs=1,
        )

        metricas_medias = {}
        for nome_metrica in scoring:
            metricas_medias[nome_metrica] = float(
                resultados_cv[f"test_{nome_metrica}"].mean()
            )

        return metricas_medias

    def executar(self):
        """
        Executa o pipeline completo:

        - Prepara dados
        - Treina modelos (sklearn + MLP)
        - Avalia resultados
        - Registra no MLflow
        """

        self.logger.info("Iniciando execução do pipeline")

        # -------------------------
        # 1. Preparação dos dados
        # -------------------------
        self.data_prep.preparar()
        self.logger.info("Dados preparados com sucesso")

        # -------------------------
        # 2. Modelos
        # -------------------------
        modelos_a_treinar = self.model_factory.criar_modelos()

        self.logger.info("Modelos definidos (sklearn + MLP)")

        # -------------------------
        # 3. Treinamento
        # -------------------------
        modelos_treinados = self.treinador.treinar_modelos(
            modelos_a_treinar,
            self.data_prep.X_train,
            self.data_prep.y_train,
        )

        self.logger.info("Treinamento concluído")

        # -------------------------
        # 4. Avaliação + MLflow
        # -------------------------
        resultados = []

        mlflow.set_experiment("churn_baseline")

        for nome, modelo in modelos_treinados.items():
            self.logger.info(f"Avaliando modelo: {nome}")

            with mlflow.start_run(run_name=nome):
                metricas_cv = self._calcular_metricas_cv(
                    modelos_a_treinar[nome],
                    self.data_prep.X_train,
                    self.data_prep.y_train,
                )

                # -------------------------
                # Previsões (compatível sklearn + PyTorch)
                # -------------------------
                y_pred = self.treinador.predict(modelo, self.data_prep.X_test)

                y_prob = self.treinador.predict_proba(modelo, self.data_prep.X_test)

                # -------------------------
                # Métricas
                # -------------------------
                resultado = self.avaliador.calcular_metricas(
                    self.data_prep.y_test,
                    y_pred,
                    y_prob,
                )

                custo = self.avaliador.calcular_custo_negocio(
                    self.data_prep.y_test,
                    y_pred,
                )

                # -------------------------
                # MLflow
                # -------------------------
                mlflow.log_param("modelo", nome)
                mlflow.log_param("dataset", "dados_limpos")

                for k, v in resultado.items():
                    if self._valor_numerico_valido(v):
                        mlflow.log_metric(k, float(v))

                for k, v in metricas_cv.items():
                    if self._valor_numerico_valido(v):
                        mlflow.log_metric(k, float(v))
                    else:
                        self.logger.warning(
                            "Métrica de CV ignorada por valor inválido: %s=%s",
                            k,
                            v,
                        )

                mlflow.log_metric("custo_negocio", float(custo))

                # Salvar modelo
                mlflow.sklearn.log_model(
                    sk_model=modelo,
                    name="model",
                    registered_model_name=f"churn_{nome.lower()}",
                )

                # -------------------------
                # Resultado para tabela
                # -------------------------
                resultados.append(
                    {
                        "modelo": nome,
                        **resultado,
                        **{
                            k: v
                            for k, v in metricas_cv.items()
                            if self._valor_numerico_valido(v)
                        },
                        "custo_negocio": custo,
                    }
                )

        self.logger.info("Avaliação concluída")

        # -------------------------
        # 5. Consolidar resultados
        # -------------------------
        df_resultados = pd.DataFrame(resultados)

        df_resultados = df_resultados.sort_values(
            by="modelo",
            ascending=True,
        )

        self.logger.info("Tabela final gerada com sucesso")

        return df_resultados
