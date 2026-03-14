import logging
from pathlib import Path

from src.models.EDA import EDA
from src.models.treino import Treino
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

    # # Relatorio para validar os dados normalizados
    # base_de_relatorio = pd.concat(
    #     [variaveis_explicaveis, pd.DataFrame(variavel_target,
    # columns=[COLUNA_TARGET])],
    #     axis=1
    # )
    # Relatorio.criar_histograma(base_de_relatorio, COLUNA_TARGET)

    treino = Treino(variaveis_explicaveis, variavel_target)

    treino.split_dados()

    modelos = treino.criar_modelos()

    resultados = treino.avaliar_modelos(modelos)
    LOGGER.info("Resultados dos modelos:\n%s",
                resultados.to_string(index=False))

    melhor_modelo = resultados.iloc[0]["modelo"]

    # plotar árvore
    # treino.plotar_arvore_decisao(modelos["decision_tree"])

    # salvar melhor modelo
    Treino.salvar_modelo(
        modelos[melhor_modelo],
        melhor_modelo,
        str(ROOT_DIR / "models/trained_models"),
    )
