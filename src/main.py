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
    MLFLOW_DB_PATH = ROOT_DIR / "mlflow.db"
    MLFLOW_ARTIFACTS_PATH = ROOT_DIR / "mlruns"

    DADOS_PATH = ROOT_DIR / "data/processed/teste-2-variaveis_explicaveis.csv"
    RESULTADOS_PATH = ROOT_DIR / "report" / "publicacao_modelo_dev"

    RESULTADOS_PATH.mkdir(parents=True, exist_ok=True)
    MLFLOW_ARTIFACTS_PATH.mkdir(parents=True, exist_ok=True)

    # Carregar os dados
    pd_dados = Arquivo.carregar_dados(str(DADOS_PATH))

    # Corrige qualidade dos dados sem aplicar transformacoes que geram leakage
    normalizar = EDA()

    variaveis_explicaveis, variavel_target = normalizar.split_dados(
        dados=pd_dados
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

    modelos = treino.criar_modelos(max_depth=2, kn_neighbors= 3 )

    resultados = treino.avaliar_modelos(modelos)

    resultados.to_csv(
        RESULTADOS_PATH / "resultados_modelos_fase1.csv",
        index=False,
    )

    LOGGER.info("Resultados dos modelos:\n%s",
                resultados.to_string(index=False))

    treino.registrar_experimentos_mlflow(
        modelos=modelos,
        resultados=resultados,
        dataset_path=str(DADOS_PATH),
        nome_experimento="telco_churn_fase1",
        tracking_uri=f"sqlite:///{MLFLOW_DB_PATH}",
    )

    melhor_modelo = resultados.iloc[0]["modelo"]

    # plotar árvore
    # treino.plotar_arvore_decisao(modelos["decision_tree"])

    # salvar melhor modelo
    Treino.salvar_modelo(
        modelos[melhor_modelo],
        melhor_modelo,
        str(ROOT_DIR / "models/trained_models"),
    )
