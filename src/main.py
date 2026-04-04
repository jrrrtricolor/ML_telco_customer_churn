#Bibliotecas internas.
import src.pipeline as pipeline
import logging
# IMPORTAR SUA CONFIG
from src.config.logging_config import setup_logging


if __name__ == "__main__":
    # ATIVAR LOGGING 🔥
    setup_logging()

    logger = logging.getLogger(__name__)
    # Criar uma instância da pipeline
    minha_pipeline = pipeline.Pipeline()

    # Executar a pipeline
    resultados = minha_pipeline.executar()

    # Exibir os resultados
    logger.info("\n%s", resultados)