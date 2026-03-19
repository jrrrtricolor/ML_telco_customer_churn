import logging
from pathlib import Path
import pandas as pd

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
    RESULTADOS_PATH = ROOT_DIR / "report" / "publicacao_modelo_dev"

    RESULTADOS_PATH.mkdir(parents=True, exist_ok=True)

    # Carregar os dados
    pd_dados = Arquivo.carregar_dados(str(DADOS_PATH))

    # Corrige qualidade dos dados sem aplicar transformacoes que geram leakage
    normalizar = EDA()
    
    variaveis_explicaveis, variavel_target = normalizar.normalizar_dados(
        dados=pd_dados
    )

    Arquivo.salvar_dados(
        pd.concat([variaveis_explicaveis, pd.DataFrame(variavel_target, columns=["Churn"])], axis=1),
        "teste-2-variaveis_explicaveis.csv",
        str(ROOT_DIR / "data/processed"),
        "csv",
    )
