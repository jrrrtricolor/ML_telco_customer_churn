import logging
import pandas as pd
import mlflow.sklearn

# Bibliotecas internas
from src.load import DataLoader
from src.data_prep import DataPreprocessor
from src.model_factory import ModelFactory
from src.avaliador import Avaliador
from src.trainer import Trainer


# Usa banco SQLite (mais estável)
TRACKING_URI = 'sqlite:///mlflow.db'

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_registry_uri(TRACKING_URI)

class Pipeline:
    def __init__(self):
        self.data_loader = DataLoader(data_path="data/raw/Telco_Customer_Churn.csv")
        self.dataprep = DataPreprocessor(pd_dataframe=self.data_loader.data, seed=2711, test_size=0.2)
        self.avaliador = Avaliador()
        self.logger = logging.getLogger(__name__)
        self.treinador = Trainer()
        self.modelfactory = ModelFactory(seed=2711)

    def executar(self):
        # Preparar dados
        # TODO: Implementar dentro a método normalizar função que deixe os valores entre 0  e 1 para
        # evitar a disparidade entre os valores das features, o que
        # pode afetar o desempenho dos modelos de machine learning.
        self.dataprep.preparar()

        # Lista de modelos a serem treinados.
        modelos_a_treinar = self.modelfactory.criar_modelos()

        # Treinar modelos
        modelos_treinados = self.treinador.treinar_modelos(
            modelos_a_treinar, self.dataprep.X_train, self.dataprep.y_train
        )

        # Avaliar modelos
        resultados = []
        # Definindo o nome do experimento no MLflow
        mlflow.set_experiment("churn_baseline")

        for nome, modelo in modelos_treinados.items():
            # Iniciar uma nova execução no MLflow para cada modelo
            with mlflow.start_run(run_name=nome):

                resultado = self.avaliador.avaliar(
                    modelo,
                    self.dataprep.X_test,
                    self.dataprep.y_test,
                )

                # -------------------------
                # LOG NO MLFLOW
                # -------------------------

                # Parâmetros do modelo
                mlflow.log_param("modelo", nome)

                # Métricas (somente números)
                for k, v in resultado.items():
                    if isinstance(v, (int, float)):
                        mlflow.log_metric(k, v)

                # Métrica de negócio
                y_pred = modelo.predict(self.dataprep.X_test)
                mlflow.log_metric("custo_negocio",
                                  self.avaliador.calcular_custo_negocio(self.dataprep.y_test,
                                                                        y_pred))

                # Dataset (versão simples)
                mlflow.log_param("dataset", "dados_limpos")

                # Salvar modelo
                mlflow.sklearn.log_model(
                    sk_model=modelo,
                    name="model",
                    registered_model_name=f"churn_{nome.lower()}"
                )
            # Salvandos os dados para ter um dataframe para exibir no final da execução
            resultados.append({
                "modelo": nome,
                **resultado
            })

        # Consolidar resultados
        df_resultados = pd.DataFrame(resultados)

        # Ordenar pelo nome do modelo
        df_resultados = df_resultados.sort_values(by="modelo", ascending=True)

        return df_resultados
