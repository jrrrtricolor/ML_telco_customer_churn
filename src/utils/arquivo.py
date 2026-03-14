import os
import pandas as pd
from numpy.f2py.auxfuncs import throw_error


# classe responsavel por métodos que vão lidar com arquivos, como carregar os dados, salvar os dados, etc.

class Arquivo:
    def __init__(self):
        pass

    @staticmethod
    def carregar_dados(path: str) -> pd.DataFrame:
        # Carregar os dados
        if os.path.exists(path):
            dados = pd.read_csv(path)
            return dados
        else:
            raise FileNotFoundError(f"Arquivo {path} não encontrado.")

    @staticmethod
    def salvar_dados(dados: pd.DataFrame, file_name: str, path: str, tipo_arquivo) -> None:
        # Salvar os dados em um arquivo.
        file_path = os.path.join(path, file_name)
        try:
            if tipo_arquivo == 'csv':
                dados.to_csv(file_path, index=False)
            elif tipo_arquivo == 'xlsx':
                dados.to_excel(file_path, index=False)
        except Exception as e:
            raise IOError(f"Erro ao salvar o arquivo: {e}")

        print(f"Arquivo salvo com sucesso em: {file_path}")
