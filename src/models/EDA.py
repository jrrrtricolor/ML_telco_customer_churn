import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


class EDA:
    def __init__(self, dados):
        self.dados = dados

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

    def normalizar_dados(
        self,
        colunas_a_remover: list[str],
        coluna_target: str,
    ) -> tuple[pd.DataFrame, np.ndarray]:
        # Separar as variáveis explicativas e a variável target
        dados = self.dados.drop(columns=colunas_a_remover).copy()
        variaveis_explicaveis = dados.drop(columns=[coluna_target]).copy()

        # Transformando a variavel alvo.
        label_encoder = LabelEncoder()
        variavel_target = pd.Series(
            label_encoder.fit_transform(dados[coluna_target]),
            index=dados.index,
            name=coluna_target,
        )

        # Corrigir os valores da coluna TotalCharges
        variaveis_explicaveis = self.corrigir_valores_total_charges(
            variaveis_explicaveis)

        # Corrigir valores numéricos sem aplicar escala global.
        # A escala sera feita no pipeline, apos o split, para evitar leakage.
        colunas_numericas = [
            "tenure",
            "MonthlyCharges",
            "TotalCharges",
        ]
        variaveis_explicaveis = self.corrigir_valores_numericos(
            variaveis_explicaveis,
            colunas_numericas,
        )

        # Mantem o target alinhado com possiveis drops de linhas invalidas.
        variavel_target = variavel_target.loc[variaveis_explicaveis.index]

        return variaveis_explicaveis, variavel_target.to_numpy()
