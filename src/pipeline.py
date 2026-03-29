import logging
import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.pytorch
import torch

# Bibliotecas internas
from src.load import DataLoader
from src.data_prep import DataPreprocessor
from src.model_factory import ModelFactory
from src.avaliador import Avaliador
from src.trainer import Trainer
from src.mlp_model import MLPModel


# Usa banco SQLite (mais estável)
TRACKING_URI = "sqlite:///mlflow.db"

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_registry_uri(TRACKING_URI)


class Pipeline:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

        self.data_loader = DataLoader(
            data_path="data/raw/Telco_Customer_Churn.csv"
        )

        self.dataprep = DataPreprocessor(
            pd_dataframe=self.data_loader.data,
            seed=2711,
            test_size=0.2,
        )

        self.avaliador = Avaliador()
        self.treinador = Trainer()
        self.modelfactory = ModelFactory(seed=2711)

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
        self.dataprep.preparar()
        self.logger.info("Dados preparados com sucesso")

        # -------------------------
        # 2. Modelos
        # -------------------------
        modelos_a_treinar = self.modelfactory.criar_modelos()

        # Adiciona MLP (PyTorch)
        input_size = self.dataprep.X_train.shape[1]
        modelos_a_treinar["MLP"] = MLPModel(input_size=input_size)

        self.logger.info("Modelos definidos (sklearn + MLP)")

        # -------------------------
        # 3. Treinamento
        # -------------------------
        modelos_treinados = self.treinador.treinar_modelos(
            modelos_a_treinar,
            self.dataprep.X_train,
            self.dataprep.y_train,
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

                # -------------------------
                # Previsões (compatível sklearn + PyTorch)
                # -------------------------
                y_pred = self.treinador.predict(
                    modelo, self.dataprep.X_test
                )

                y_prob = self.treinador.predict_proba(
                    modelo, self.dataprep.X_test
                )

                # -------------------------
                # Métricas
                # -------------------------
                resultado = self.avaliador.calcular_metricas(
                    self.dataprep.y_test,
                    y_pred,
                    y_prob,
                )

                custo = self.avaliador.calcular_custo_negocio(
                    self.dataprep.y_test,
                    y_pred,
                )

                # -------------------------
                # MLflow
                # -------------------------
                mlflow.log_param("modelo", nome)
                mlflow.log_param("dataset", "dados_limpos")

                for k, v in resultado.items():
                    if isinstance(v, (int, float)):
                        mlflow.log_metric(k, float(v))

                mlflow.log_metric("custo_negocio", float(custo))

                # Salvar modelo
                if isinstance(modelo, torch.nn.Module):
                    mlflow.pytorch.log_model(
                        modelo,
                        name="model",
                        registered_model_name=f"churn_{nome.lower()}",
                    )
                else:
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