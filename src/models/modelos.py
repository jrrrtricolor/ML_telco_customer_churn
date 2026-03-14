import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


class Modelos:
    @staticmethod
    def criar_preprocessador(variaveis_explicaveis: pd.DataFrame) -> ColumnTransformer:
        colunas_numericas = variaveis_explicaveis.select_dtypes(include=["number"]).columns.tolist()
        colunas_categoricas = variaveis_explicaveis.select_dtypes(exclude=["number"]).columns.tolist()

        transformadores = []

        if colunas_numericas:
            transformadores.append(("numericas", MinMaxScaler(), colunas_numericas))

        if colunas_categoricas:
            transformadores.append(
                (
                    "categoricas",
                    OneHotEncoder(drop="if_binary", handle_unknown="ignore", sparse_output=False),
                    colunas_categoricas,
                )
            )

        if not transformadores:
            raise ValueError("Nenhuma coluna disponível para preprocessamento.")

        return ColumnTransformer(transformers=transformadores, remainder="drop")

    @staticmethod
    def criar_pipeline(modelo, variaveis_explicaveis: pd.DataFrame) -> Pipeline:
        preprocessador = Modelos.criar_preprocessador(variaveis_explicaveis)

        return Pipeline(
            steps=[
                ("preprocessador", preprocessador),
                ("modelo", modelo),
            ]
        )
