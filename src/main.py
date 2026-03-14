import pandas as pd

from src.utils.arquivo import Arquivo
from src.models.EDA import EDA
from report.relatorio import Relatorio
from src.models.treino import Treino

if __name__ == "__main__":
    DADOS_PATH = '../data/raw/Telco_Customer_Churn.csv'
    COLUNA_TARGET = "Churn"

    # Carregar os dados
    pd_dados = Arquivo.carregar_dados(DADOS_PATH)

    # Normalizando os dados
    normalizar = EDA(dados=pd_dados)
    colunas_remover = ['customerID']

    variaveis_explicaveis, variavel_target = normalizar.normalizar_dados(colunas_a_remover=colunas_remover,
                                                                         coluna_target=COLUNA_TARGET)
    # Salvar os dados normalizados
    Arquivo.salvar_dados(variaveis_explicaveis
                         , 'teste-1-variaveis_explicaveis.csv'
                         , '../data/processed'
                         ,'csv')

    # # Relatorio para validar os dados normalizados
    # base_de_relatorio = pd.concat(
    #     [variaveis_explicaveis, pd.DataFrame(variavel_target, columns=[COLUNA_TARGET])],
    #     axis=1
    # )
    # Relatorio.criar_histograma(base_de_relatorio, COLUNA_TARGET)

    treino = Treino(variaveis_explicaveis, variavel_target)

    treino.split_dados()

    modelos = treino.criar_modelos()

    resultados = treino.avaliar_modelos(modelos)

    print(resultados)

    # plotar árvore
    # treino.plotar_arvore_decisao(modelos["decision_tree"])

    # salvar melhor modelo
    Treino.salvar_modelo(modelos["random_forest"], "random_forest")
