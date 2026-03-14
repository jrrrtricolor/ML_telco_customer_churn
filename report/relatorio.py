import pandas as pd
import plotly.express as py


class Report:

    def __init__(self, arquivo: [str]):
        self.arquivo = arquivo


# Método para criar histograma com a finalidade de validar ou enteder a qualidade dos dados
def criar_histograma(dataset: pd.DataFrame, feature: str, feature_em_foco: str) -> None:
    print('#' * 50, f'Histograma da coluna: {feature}', '#' * 50)

    if feature in dataset.columns:
        grafico = py.histogram(dataset
                               , x=feature
                               , text_auto=True
                               , title=f'Distribuição de {feature}'
                               , color=feature_em_foco
                               , barmode='group')
        grafico.show()
