# Bibliotecas internas.
import logging

import src.pipeline as pipeline
from src.config.logging_config import setup_logging

if __name__ == "__main__":
    # Ativa a configuração de logging.
    setup_logging()

    logger = logging.getLogger(__name__)

    # Cria uma instância da pipeline.
    minha_pipeline = pipeline.Pipeline()

    # Executa a pipeline.
    resultados = minha_pipeline.executar()

    # Exibe os resultados no log.
    logger.info("\n%s", resultados)