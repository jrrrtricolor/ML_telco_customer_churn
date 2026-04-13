import logging

import pandas as pd

# Bibliotecas internas
from src.utils import Utilidades


class DataLoader:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.extensao_arquivo = self.data_path.split(".")[-1]
        self.data = None
        self.logger = logging.getLogger(__name__)
        self.utilidades = Utilidades()
        self._carregar_dados()

    def _carregar_dados(self) -> pd.DataFrame:
        # Carrega os dados do arquivo com pandas.
        try:
            if self.extensao_arquivo == "csv":
                self.data = pd.read_csv(self.data_path)
            else:
                self.logger.error("Extensão de arquivo não suportada.")
                raise ValueError("Extensão não suportada!")

        except Exception as e:
            self.logger.error(f"Erro ao carregar dados: {e}")
            raise e

        # Valida os dados carregados.
        self.utilidades.validar_dados(self.data)

        return self.data
