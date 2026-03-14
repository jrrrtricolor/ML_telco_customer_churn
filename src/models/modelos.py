import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder

class Modelos:
    def __init__(self, modelo, tokenizer):
        self.modelo = modelo

    @staticmethod
    def criar_one_hot_model(variaveis_explicaveis: pd.DataFrame) -> pd.DataFrame:
        # Aplicando o one hot encoding nas variáveis explicativas. Basicamente transforma
        # as variáveis categóricas em variáveis numéricas, criando uma nova coluna para cada
        # categoria e atribuindo um valor binário (0 ou 1) para indicar a presença ou ausência
        # da categoria em cada linha.

        # Resultado é o modelo que transforma registros em um formato adequado para algoritmos
        # de machine learning, que geralmente requerem dados numéricos.

        # Definir as colunas para o one hot encoding
        colunas_para_hot_encode = variaveis_explicaveis.columns

        one_hot = make_column_transformer(
            (OneHotEncoder(drop='if_binary')
                 , variaveis_explicaveis.select_dtypes(include=["object", "string"]).columns)
            , remainder='passthrough'
            , sparse_threshold=0
        )

        variaveis_explicaveis = one_hot.fit_transform(variaveis_explicaveis)

        # Criar um DataFrame com os nomes das colunas gerados pelo one hot encoding
        variaveis_explicaveis = pd.DataFrame(variaveis_explicaveis,
                                             columns=one_hot.get_feature_names_out(colunas_para_hot_encode))

        return variaveis_explicaveis
