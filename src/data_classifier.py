import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

class DataClassifier:
    def __init__(self):
        self.one_hot_encoders = {}
        self.label_encoders = {}

    def fit(self, X, y):
        # Copiar os dados para realizar o encoding
        dados_encoded = X

        # Iterar sobre as colunas do DataFrame para identificar as colunas categóricas e aplicar o encoding adequado
        for column in dados_encoded.columns:
            # Verificar se a coluna é do tipo string (categórica)
            if dados_encoded[column].dtype == "object":
                quantidade_valores_unicos = dados_encoded[column].nunique()

                # Caso a feature em questão seja binária, aplicar o LabelEncoder. Caso contrário, aplicar o OneHotEncoder.
                if quantidade_valores_unicos == 2:
                    self.label_encoders[column] = LabelEncoder()
                    # binária → LabelEncoder
                    self.label_encoders[column].fit(
                        dados_encoded[column]
                    )

                else:
                    # categórica → OneHotEncoder
                    # Ou seja, transfomar as categorias em colunas binárias.
                    encoder = OneHotEncoder(
                        sparse_output=False, handle_unknown="ignore"
                    )

                    self.one_hot_encoders[column] = encoder
                    encoder.fit(dados_encoded[[column]])

        return self

    def transform(self, X):
        # Copiar os dados para realizar o encoding
        dados_encoded = X.copy()

        colunas_para_remover = []
        dfs_encoded = []

        # Iterar sobre as colunas do DataFrame para identificar as colunas categóricas e aplicar o encoding adequado
        for column in dados_encoded.columns:
            # Verificar se a coluna é do tipo string (categórica)
            if self.label_encoders.get(column) is not None:
                quantidade_valores_unicos = dados_encoded[column].nunique()

                # Caso a feature em questão seja binária, aplicar o LabelEncoder. Caso contrário, aplicar o OneHotEncoder.
                if quantidade_valores_unicos <= 2:
                    label_encoder = self.label_encoders[column]

                    dados_encoded[column] = label_encoder.transform(
                        dados_encoded[column]
                    )
                else:
                    raise ValueError(f'A coluna "{column}" não é binária, mas possui um LabelEncoder associado. Verifique os encoders e os dados de entrada.')

            if self.one_hot_encoders.get(column) is not None:
                encoder = self.one_hot_encoders[column]

                encoded = encoder.transform(dados_encoded[[column]])
                colunas = encoder.get_feature_names_out([column])

                df_temp = pd.DataFrame(
                    encoded, columns=colunas, index=dados_encoded.index
                )

                dfs_encoded.append(df_temp)
                colunas_para_remover.append(column)

        # Criando um novo pandas DataFrame concatenando os dados originais (com as colunas categóricas removidas)
        # e os dados codificados (colunas binárias criadas a partir do OneHotEncoder).
        return pd.concat(
            [dados_encoded.drop(columns=colunas_para_remover)] + dfs_encoded, axis=1
        )
