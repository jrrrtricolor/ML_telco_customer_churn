"""Gera relatório automático após commit."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from src.models.treino import treinar_modelo_baseline
from src.models.treino_modelos_arvore import treinar_e_comparar_modelos

METAS_MINIMAS = {
    "roc_auc": 0.75,
    "pr_auc": 0.55,
    "recall": 0.70,
    "precision": 0.45,
    "f1": 0.58,
    "accuracy": 0.70,
}


def obter_raiz_projeto() -> Path:
    """Retorna a raiz do repositório."""
    return Path(__file__).resolve().parents[2]


def formatar_metricas(metricas: dict[str, float]) -> str:
    """Converte dicionário de métricas em markdown."""
    linhas = []
    for nome in ["roc_auc", "pr_auc", "recall", "precision", "f1", "accuracy"]:
        valor = metricas[nome]
        meta = METAS_MINIMAS[nome]
        status = "ATINGIU" if valor >= meta else "NAO_ATINGIU"
        linhas.append(f"- {nome}: {valor:.4f} (meta {meta:.2f}) -> {status}")
    return "\n".join(linhas)


def gerar_recomendacoes(metricas_modelo: dict[str, float]) -> list[str]:
    """Gera recomendações simples com base nas métricas do modelo recomendado."""
    recomendacoes = []
    if metricas_modelo["recall"] < 0.75:
        recomendacoes.append(
            "Aumentar recall: avaliar ajuste de limiar para capturar mais churners."
        )
    if metricas_modelo["precision"] < 0.52:
        recomendacoes.append(
            "Melhorar precision: revisar features e calibrar probabilidade para reduzir falso positivo."
        )
    if metricas_modelo["pr_auc"] < 0.60:
        recomendacoes.append(
            "Subir PR AUC: testar engenharia de atributos orientada a sinais de retenção."
        )
    if not recomendacoes:
        recomendacoes.append(
            "Modelo está estável: seguir para etapa de API, monitoramento e deploy."
        )
    return recomendacoes


def gerar_grafico_comparacao_modelos(
    modelos: dict[str, dict[str, Any]],
    pasta_saida: Path,
) -> Path:
    """Gera gráfico de barras com ROC AUC, PR AUC e Recall por modelo."""
    linhas = []
    for nome, dados in modelos.items():
        metrica = dados["metricas_teste"]
        linhas.append(
            {
                "modelo": nome,
                "roc_auc": metrica["roc_auc"],
                "pr_auc": metrica["pr_auc"],
                "recall": metrica["recall"],
            }
        )

    tabela = pd.DataFrame(linhas).set_index("modelo")
    caminho = pasta_saida / "comparacao_modelos.png"

    figura, eixo = plt.subplots(figsize=(10, 5))
    tabela.plot(kind="bar", ax=eixo)
    eixo.set_title("Comparação de modelos no teste")
    eixo.set_ylabel("Score")
    eixo.set_xlabel("Modelo")
    eixo.set_ylim(0, 1)
    eixo.legend(title="Métrica")
    figura.tight_layout()
    figura.savefig(caminho, dpi=140)
    plt.close(figura)
    return caminho


def gerar_grafico_gap_metas(
    metricas_modelo: dict[str, float],
    pasta_saida: Path,
) -> Path:
    """Gera gráfico comparando resultado do modelo recomendado vs meta mínima."""
    ordem = ["roc_auc", "pr_auc", "recall", "precision", "f1", "accuracy"]
    valores = [metricas_modelo[nome] for nome in ordem]
    metas = [METAS_MINIMAS[nome] for nome in ordem]
    caminho = pasta_saida / "gap_metas_modelo_recomendado.png"

    figura, eixo = plt.subplots(figsize=(10, 5))
    eixo.plot(ordem, valores, marker="o", label="Resultado")
    eixo.plot(ordem, metas, marker="o", label="Meta mínima")
    eixo.set_title("Modelo recomendado vs metas mínimas")
    eixo.set_ylabel("Score")
    eixo.set_ylim(0, 1)
    eixo.grid(axis="y", alpha=0.3)
    eixo.legend()
    figura.tight_layout()
    figura.savefig(caminho, dpi=140)
    plt.close(figura)
    return caminho


def montar_proximos_passos() -> list[str]:
    """Retorna próximos passos objetivos por prioridade."""
    return [
        "Implementar API FastAPI com /health e /predict + testes de contrato.",
        "Integrar MLflow para rastrear params, métricas e artefatos de todos os treinos.",
        "Containerizar treino e API com Dockerfile dedicado para cada fluxo.",
        "Subir monitoramento no Databricks Free com job diário e alertas de queda de recall.",
        "Adicionar modelo PyTorch tabular e comparar com as árvores no mesmo protocolo.",
    ]


def montar_previsao_conclusao() -> list[tuple[str, str]]:
    """Monta previsão simples de conclusão por fase."""
    return [
        ("Fase 1 - API e testes", "2 semanas"),
        ("Fase 2 - MLflow e monitoramento", "2 semanas"),
        ("Fase 3 - Docker e deploy cloud", "2 semanas"),
        ("Fase 4 - PyTorch e hardening", "2 semanas"),
        ("Previsão total", "8 semanas"),
    ]


def main() -> None:
    """Executa treino rápido e salva relatório em report/."""
    _, metricas_baseline = treinar_modelo_baseline(
        caminho_dados="data/processed/telco_churn_encoded.csv"
    )
    comparacao = treinar_e_comparar_modelos(
        caminho_dados="data/processed/telco_churn_encoded.csv",
        usar_grade_reduzida=True,
        n_splits_cv=3,
    )

    nome_recomendado = comparacao["modelo_recomendado_retencao"]
    metricas_recomendado = comparacao["modelos"][nome_recomendado]["metricas_teste"]
    recomendacoes = gerar_recomendacoes(metricas_recomendado)
    proximos_passos = montar_proximos_passos()
    previsao_conclusao = montar_previsao_conclusao()
    data_execucao = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    pasta_relatorio = obter_raiz_projeto() / "report"
    pasta_relatorio.mkdir(parents=True, exist_ok=True)
    caminho_grafico_comparacao = gerar_grafico_comparacao_modelos(
        comparacao["modelos"], pasta_relatorio
    )
    caminho_grafico_gap = gerar_grafico_gap_metas(metricas_recomendado, pasta_relatorio)

    conteudo = f"""# Relatório Automático do Commit

Data de execução: {data_execucao}

## Resumo
- Melhor ROC AUC: {comparacao["melhor_modelo"]}
- Modelo recomendado para retenção: {nome_recomendado}

## Métricas do baseline (regressão logística)
{formatar_metricas(metricas_baseline)}

## Métricas do modelo recomendado
{formatar_metricas(metricas_recomendado)}

## Gráficos de apoio
![Comparação de modelos]({caminho_grafico_comparacao.name})

![Gap de metas]({caminho_grafico_gap.name})

## Recomendações
{chr(10).join(f"- {item}" for item in recomendacoes)}

## Próximos passos
{chr(10).join(f"- {item}" for item in proximos_passos)}

## Previsão de conclusão do projeto
{chr(10).join(f"- {fase}: {prazo}" for fase, prazo in previsao_conclusao)}
"""

    caminho_relatorio = pasta_relatorio / "relatorio_commit.md"
    caminho_relatorio.write_text(conteudo, encoding="utf-8")
    print(f"Relatório gerado em: {caminho_relatorio}")


if __name__ == "__main__":
    main()
