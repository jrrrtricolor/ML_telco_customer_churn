import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


class Relatorio:

    @staticmethod
    def salvar_grafico(fig: go.Figure, file_name: str, path: str, tipo_arquivo: str) -> None:

        file_path = os.path.join(path, file_name)

        try:
            if tipo_arquivo in ["png", "jpg"]:
                fig.write_image(file_path)

            elif tipo_arquivo == "html":
                fig.write_html(file_path)

            else:
                raise ValueError("Tipo de arquivo não suportado")

        except Exception as e:
            raise IOError(f"Erro ao salvar o gráfico: {e}")

        print(f"Gráfico salvo com sucesso em: {file_path}")

    @staticmethod
    def criar_histograma(dataset: pd.DataFrame, feature_em_foco: str) -> None:

        for feature in dataset.columns:
            print('#' * 50, f'Histograma da coluna: {feature}', '#' * 50)

            grafico = px.histogram(
                dataset,
                x=feature,
                text_auto=True,
                title=f'Distribuição de {feature}',
                color=feature_em_foco,
                barmode='group'
            )

            Relatorio.salvar_grafico(
                grafico,
                file_name=f"{feature}-to-{feature_em_foco}-histograma.png",
                path="../report/normalizado",
                tipo_arquivo="png"
            )

