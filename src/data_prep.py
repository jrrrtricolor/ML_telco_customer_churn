import logging

import pandas as pd
from sklearn.model_selection import train_test_split

# Bibliotecas internas
from src.utils import Utilidades


class DataPreprocessor:
    # Classe responsável por preparar os dados para os treinamentos dos modelos de machine learning.

    def __init__(self, pd_dataframe: pd.DataFrame, seed: int = 42, test_size: float = 0.2):
        self.logger = logging.getLogger(__name__)
        self.dados = pd_dataframe
        self.X = None  # Atributo (features) irá conter as colunas explicativas.
        self.y = None  # Atributo (target) irá conter a coluna resposta.
        # Atributos para armazenar os dados de treino e teste
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        # Definir SEED para garantir a reprodutibilidade dos resultados
        self.SEED = seed

        # Definir o tamanho do conjunto de teste
        self.TEST_SIZE = test_size

        # Importar classes de utilidades
        self.utilidades = Utilidades

    def converter_features_numericas(self, feature_names: list[str]):
        # Converter as features numéricas para o tipo float
        for feature_name in feature_names:
            try:
                self.dados[feature_name] = pd.to_numeric(self.dados[feature_name], errors="coerce")
            except Exception as e:
                self.logger.error(
                    f"Problemas ao tentar converter a coluna '{feature_name}' para o tipo float: {e}"
                )
                raise e

        self.logger.info(
            f"Coluna '{feature_names}' convertida para o tipo float com sucesso."
        )

    def separar_features_target(self, target_column: str):
        # Separar as features do target
        try:
            self.X = self.dados.drop(target_column, axis=1)
            self.y = self.dados[target_column]
        except Exception as e:
            self.logger.error(f"Problemas ao tentar separar as features do target: {e}")
            raise e

    def dividir_treino_teste(self):
        # Dividir os dados em treino e teste mantendo a proporção da classe alvo.

        try:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X,
                self.y,
                test_size=self.TEST_SIZE,
                random_state=self.SEED,
                stratify=self.y,
            )
        except Exception as e:
            self.logger.error(
                "Problemas ao tentar separar os dados de treino e teste", e
            )
            raise e

    def remover_outliers(self, feature_name: str):
        # Remover colunas irrelevantes
        if feature_name in self.dados.columns:
            try:
                self.dados.drop(feature_name, axis=1, inplace=True)
            except Exception as e:
                self.logger.error(
                    f"Problemas ao tentar remover a coluna '{feature_name}': {e}"
                )
                raise e

            self.logger.info(f"Coluna '{feature_name}' removida com sucesso.")

    def remover_valores_menor_igual_zero(self):

        # Remover linhas com valores mensais 0
        try:
            self.dados = self.dados[self.dados["MonthlyCharges"] > 0]
        except Exception as e:
            self.logger.error(
                f"Problemas ao tentar remover as linhas com valores mensais menor ou igual a 0: {e}"
            )
            raise e

        if self.dados["MonthlyCharges"].min() > 0:
            self.logger.info(
                "Linhas com valores menor ou igual a 0 removidas com sucesso."
            )

    def normalizar_dados(self) -> pd.DataFrame:
        # Tratamento de churn.
        self.dados["Churn"] = self.dados["Churn"].map({"Yes": 1, "No": 0})
        return self.dados

    def salvar_arquivo(self, pd_dataframe: pd.DataFrame):

        pd_dataframe.to_csv("data/processed/dados_limpos.csv", index=False)

    def preparar(self):

        # Remover colunas irrelevantes
        self.remover_outliers("customerID")

        # Remover linhas com valores mensais 0
        self.remover_valores_menor_igual_zero()

        # Converter as features numéricas para o tipo float
        self.converter_features_numericas(["TotalCharges", "MonthlyCharges", "tenure"])

        # Tratamento de valores ausentes.
        self.dados["TotalCharges"] = self.dados["TotalCharges"].fillna(
            self.dados["tenure"] * self.dados["MonthlyCharges"]
        )

        # Normalizar os dados
        dados_encoded = self.normalizar_dados()

        # Salvar os dados limpos em um arquivo CSV
        self.salvar_arquivo(dados_encoded)

        # Separando as features e o target
        self.X = self.dados.drop("Churn", axis=1)
        self.y = self.dados["Churn"]

        # Dividir os dados em treino e teste
        self.dividir_treino_teste()
