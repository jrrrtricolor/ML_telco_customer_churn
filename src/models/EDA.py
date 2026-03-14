# Importar bibliotecas
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Importar classes internas
from src.models.modelos import Modelos


class EDA:
    def __init__(self, dados):
        self.dados = dados

    def corrigir_valores_total_charges(self, dados: pd.DataFrame) -> pd.DataFrame:
        # Corrigir os valores da coluna TotalCharges
        # substituir espaços por NaN
        dados["TotalCharges"] = dados["TotalCharges"].replace(" ", np.nan)

        # converter para número
        dados["TotalCharges"] = pd.to_numeric(dados["TotalCharges"])

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

    def corrigir_distancia_valores_numericos(self, dados: pd.DataFrame, colunas_numericas_list: [str]) -> pd.DataFrame:
        # Corrigir coluna numérica

        for coluna in colunas_numericas_list:
            if coluna in dados.columns:
                dados[coluna] = pd.to_numeric(
                    dados[coluna],
                    errors="raise"
                )

            # Remover valores inválidos
            dados = dados.dropna(subset=[coluna])

        # Inicializando objeto MinMaxScaler
        scaler = MinMaxScaler()

        # Aplicando o scaler nas colunas numéricas
        dados[colunas_numericas_list] = scaler.fit_transform(
            dados[colunas_numericas_list]
        )
        return dados

    def normalizar_dados(self, colunas_a_remover: list[str], coluna_target: str) -> tuple[pd.DataFrame, np.ndarray]:
        # Separar as variáveis explicativas e a variável target
        dados = self.dados.drop(columns=colunas_a_remover)
        variaveis_explicaveis = dados.drop(columns=[coluna_target])

        # Transformando a variavel alvo.
        label_enconder = LabelEncoder()
        variavel_target = label_enconder.fit_transform(dados[coluna_target])

        # Corrigir os valores da coluna TotalCharges
        variaveis_explicaveis = self.corrigir_valores_total_charges(variaveis_explicaveis)

        # Corrigir a distância entre os valores numéricos
        colunas_numericas = [
            "tenure",
            "MonthlyCharges",
            "TotalCharges",
        ]
        variaveis_explicaveis = self.corrigir_distancia_valores_numericos(variaveis_explicaveis, colunas_numericas)

        # Criar o modelo de one hot encoding e transformar as variáveis explicativas
        variaveis_explicaveis = Modelos.criar_one_hot_model(variaveis_explicaveis=variaveis_explicaveis)

        return variaveis_explicaveis, variavel_target
