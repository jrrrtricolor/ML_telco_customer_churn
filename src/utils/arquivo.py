import logging
import os
from typing import Literal

import pandas as pd

LOGGER = logging.getLogger(__name__)

class Arquivo:
    @staticmethod
    def carregar_dados(path: str) -> pd.DataFrame:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Arquivo {path} não encontrado.")

        dados = pd.read_csv(path)
        LOGGER.info("Dados carregados de: %s", path)
        return dados

    @staticmethod
    def salvar_dados(
        dados: pd.DataFrame,
        file_name: str,
        path: str,
        tipo_arquivo: Literal["csv", "xlsx"],
    ) -> None:
        file_path = os.path.join(path, file_name)
        os.makedirs(path, exist_ok=True)

        try:
            if tipo_arquivo == "csv":
                dados.to_csv(file_path, index=False)
            elif tipo_arquivo == "xlsx":
                dados.to_excel(file_path, index=False)
            else:
                raise ValueError("Tipo de arquivo não suportado.")
        except Exception as exc:
            raise OSError(f"Erro ao salvar o arquivo: {exc}") from exc

        LOGGER.info("Arquivo salvo com sucesso em: %s", file_path)
