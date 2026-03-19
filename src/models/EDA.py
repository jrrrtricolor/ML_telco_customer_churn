import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


class EDA:
    def __init__(self):
        self.colunas_a_remover = ["customerID"]
        self.coluna_target = "Churn"
        self.label_encoder = LabelEncoder()
        self.colunas_numericas = [
            "tenure",
            "MonthlyCharges",
            "TotalCharges",
        ]

    @staticmethod
    def corrigir_valores_total_charges(dados: pd.DataFrame) -> pd.DataFrame:
        # Corrigir os valores da coluna TotalCharges
        dados = dados.copy()

        # substituir espaços por NaN
        dados["TotalCharges"] = dados["TotalCharges"].replace(" ", np.nan)

        # converter para número
        dados["TotalCharges"] = pd.to_numeric(
            dados["TotalCharges"], errors="coerce")

        # encontrar valores faltantes
        mask = dados["TotalCharges"].isna()

        # corrigir tenure = 0
        tenure_ajustado = dados["tenure"].replace(0, 1)

        # calcular valor correto
        dados.loc[mask, "TotalCharges"] = (
            dados.loc[mask, "MonthlyCharges"] *
            tenure_ajustado.loc[mask]
        )

        return dados

    @staticmethod
    def corrigir_valores_numericos(
        dados: pd.DataFrame,
        colunas_numericas_list: list[str],
    ) -> pd.DataFrame:
        dados = dados.copy()
        for coluna in colunas_numericas_list:
            if coluna in dados.columns:
                dados[coluna] = pd.to_numeric(
                    dados[coluna],
                    errors="coerce",
                )

        colunas_existentes = [
            coluna for coluna in colunas_numericas_list
            if coluna in dados.columns
        ]
        if colunas_existentes:
            dados = dados.dropna(subset=colunas_existentes)

        return dados

    def processar_dados(self, x: pd.DataFrame, y: pd.Series = None) -> tuple[pd.DataFrame, np.ndarray]:
        # Corrigir os valores da coluna TotalCharges
        variaveis_explicaveis = self.corrigir_valores_total_charges(
            x)

        # Corrigir valores numéricos sem aplicar escala global.
        # A escala sera feita no pipeline, apos o split, para evitar leakage.
        variaveis_explicaveis = self.corrigir_valores_numericos(
            variaveis_explicaveis,
            self.colunas_numericas,
        )

        if y is not None:
            # Mantem o target alinhado com possiveis drops de linhas invalidas.
            variavel_target = y.loc[variaveis_explicaveis.index]

            return variaveis_explicaveis, variavel_target.to_numpy()
        
        return variaveis_explicaveis, None

    def split_dados(self, dados: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        variaveis_explicaveis = dados.drop(columns=[self.coluna_target]).copy()
        variavel_target = dados[self.coluna_target]

        return variaveis_explicaveis, variavel_target

    def normalizar_dados(
        self,
        dados: pd.DataFrame,
    ) -> tuple[pd.DataFrame, np.ndarray]:
        x, y = self.split_dados(dados.drop(columns=self.colunas_a_remover).copy())
        
        variaveis_explicaveis, variavel_target = self.processar_dados(
            x,
            pd.Series(
                self.label_encoder.fit_transform(y),
                index=x.index,
                name=self.coluna_target,
            ) 
        )

        return variaveis_explicaveis, variavel_target
