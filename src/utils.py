import pandas as pd
import logging


class Utilidades:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def validar_dados(self, pd_dataframe: pd.DataFrame):
        # Validar os dados carregados
        if pd_dataframe is None:
            message="Nenhum dado carregado para validar."
            self.logger.error(message)
            raise ValueError(message)

        if pd_dataframe.empty:
            message = "O arquivo de dados está vazio."
            self.logger.error(message)
            raise ValueError(message)

        self.logger.info("Dados validados com sucesso.")
