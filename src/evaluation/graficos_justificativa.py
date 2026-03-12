"""Geração de gráficos simples para justificar decisões de modelagem."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.data.ingestao import carregar_dados_brutos


def obter_raiz_projeto() -> Path:
    """Retorna o caminho da raiz do repositório."""
    return Path(__file__).resolve().parents[2]


def gerar_graficos_justificativa(
    caminho_dados_brutos: str | Path | None = None,
    pasta_saida: str | Path | None = None,
) -> list[Path]:
    """Gera gráficos de apoio para análise de churn e retorna caminhos gerados."""
    dataframe = carregar_dados_brutos(caminho_dados_brutos).dropna(
        subset=["TotalCharges", "Churn"]
    )
    destino = (
        Path(pasta_saida)
        if pasta_saida is not None
        else obter_raiz_projeto() / "docs" / "graficos"
    )
    destino.mkdir(parents=True, exist_ok=True)

    caminhos: list[Path] = []
    sns.set_theme(style="whitegrid")

    figura, eixo = plt.subplots(figsize=(8, 5))
    sns.countplot(data=dataframe, x="Churn", hue="Churn", ax=eixo, palette="Set2")
    eixo.set_title("Distribuição de churn no dataset")
    eixo.set_xlabel("Churn")
    eixo.set_ylabel("Quantidade de clientes")
    caminho = destino / "distribuicao_churn.png"
    figura.tight_layout()
    figura.savefig(caminho, dpi=140)
    plt.close(figura)
    caminhos.append(caminho)

    figura, eixo = plt.subplots(figsize=(10, 5))
    taxa = pd.crosstab(dataframe["Contract"], dataframe["Churn"], normalize="index")
    taxa = taxa.sort_index()
    taxa.plot(kind="bar", stacked=True, ax=eixo, colormap="viridis")
    eixo.set_title("Proporção de churn por tipo de contrato")
    eixo.set_xlabel("Tipo de contrato")
    eixo.set_ylabel("Proporção")
    eixo.legend(title="Churn")
    caminho = destino / "proporcao_churn_por_contrato.png"
    figura.tight_layout()
    figura.savefig(caminho, dpi=140)
    plt.close(figura)
    caminhos.append(caminho)

    figura, eixo = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=dataframe, x="Churn", y="tenure", hue="Churn", ax=eixo)
    eixo.set_title("Tempo de casa (tenure) por churn")
    eixo.set_xlabel("Churn")
    eixo.set_ylabel("Meses de permanência")
    caminho = destino / "tenure_por_churn.png"
    figura.tight_layout()
    figura.savefig(caminho, dpi=140)
    plt.close(figura)
    caminhos.append(caminho)

    return caminhos


def main() -> None:
    """Executa a geração dos gráficos de justificativa."""
    gerar_graficos_justificativa()


if __name__ == "__main__":
    main()
