import logging

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,  # ou INFO, dependendo do nível de detalhe desejado
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=True
    )
