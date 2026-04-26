import logging
from pathlib import Path

from src.models.EDA import EDA
from src.utils.arquivo import Arquivo

LOGGER = logging.getLogger(__name__)


def configurar_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


if __name__ == "__main__":
    configurar_logging()

    ROOT_DIR = Path(__file__).resolve().parents[1]

    DADOS_PATH = ROOT_DIR / "data/raw/Telco_Customer_Churn.csv"
    COLUNA_TARGET = "Churn"
    RESULTADOS_PATH = ROOT_DIR / "report" / "publicacao_modelo_dev"

    RESULTADOS_PATH.mkdir(parents=True, exist_ok=True)

    # Carregar os dados
    pd_dados = Arquivo.carregar_dados(str(DADOS_PATH))

    # Corrige qualidade dos dados sem aplicar transformacoes que geram leakage
    normalizar = EDA(dados=pd_dados)
    colunas_remover = ["customerID"]

    variaveis_explicaveis, variavel_target = normalizar.normalizar_dados(
        colunas_a_remover=colunas_remover,
        coluna_target=COLUNA_TARGET,
    )
    Arquivo.salvar_dados(
        variaveis_explicaveis,
        "teste-1-variaveis_explicaveis.csv",
        str(ROOT_DIR / "data/processed"),
        "csv",
    )
