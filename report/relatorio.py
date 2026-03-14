import logging
import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

LOGGER = logging.getLogger(__name__)


class Relatorio:

    @staticmethod
    def salvar_grafico(
        fig: go.Figure,
        file_name: str,
        path: str,
        tipo_arquivo: str,
    ) -> None:

        file_path = os.path.join(path, file_name)

        try:
            if tipo_arquivo in ["png", "jpg"]:
                fig.write_image(file_path)

            elif tipo_arquivo == "html":
                fig.write_html(file_path)

            else:
                raise ValueError("Tipo de arquivo não suportado")

        except Exception as exc:
            raise OSError(f"Erro ao salvar o gráfico: {exc}") from exc

        LOGGER.info("Gráfico salvo com sucesso em: %s", file_path)

    @staticmethod
    def criar_histograma(dataset: pd.DataFrame, feature_em_foco: str) -> None:

        for feature in dataset.columns:
            LOGGER.info("Gerando histograma da coluna: %s", feature)

            grafico = px.histogram(
                dataset,
                x=feature,
                text_auto=True,
                title=f"Distribuição de {feature}",
                color=feature_em_foco,
                barmode="group",
            )

            Relatorio.salvar_grafico(
                grafico,
                file_name=f"{feature}-to-{feature_em_foco}-histograma.png",
                path="../report/normalizado",
                tipo_arquivo="png",
            )
