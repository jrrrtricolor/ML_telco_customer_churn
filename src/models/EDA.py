# Importar bibliotecas
import pandas as pd
import plotly.express as py
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
import numpy as np

# Importar classes internas
from src.utils.arquivo import Arquivo
from src.models.modelos import Modelos

class Normaliza:
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

    def normalizar_dados(self) -> tuple[pd.DataFrame, np.ndarray]:



        # Separar as variáveis explicativas e a variável target
        dados = dados.drop(columns=["customerID"])
        variaveis_explicaveis = dados.drop(columns=[COLUNA_TARGET])

        # Transformando a variavel alvo.
        label_enconder = LabelEncoder()
        variavel_target = label_enconder.fit_transform(dados[COLUNA_TARGET])

        # Corrigir os valores da coluna TotalCharges
        variaveis_explicaveis = self.corrigir_valores_total_charges(variaveis_explicaveis)

        # Criar o modelo de one hot encoding e transformar as variáveis explicativas
        variaveis_explicaveis = Modelos.criar_one_hot_model(variaveis_explicaveis)

        return variaveis_explicaveis, variavel_target


if __name__ == "__main__":

    DADOS_PATH = '../../data/raw/Telco_Customer_Churn.csv'
    COLUNA_TARGET = "Churn"

    # Carregar os dados
    pd_dados = Arquivo.carregar_dados(DADOS_PATH)
    normalizar = Normalizar(dados=pd_dados)

    variaveis_explicaveis, variavel_target = normalizar.normalizar_dados()
