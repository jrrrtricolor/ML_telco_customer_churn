# Definicao de Metricas

## Objetivo

Avaliar modelos de churn considerando tanto desempenho estatistico quanto impacto financeiro das decisoes de retencao.

## Metricas tecnicas

- PR AUC: metrica principal para lidar com desbalanceamento e foco na classe churn.
- ROC AUC: capacidade geral de separacao entre churn e nao churn.
- Recall: proporcao de clientes churn corretamente identificados.
- Precision: proporcao de alertas de churn que realmente eram churn.
- F1: equilibrio entre precision e recall.
- Accuracy: metrica complementar, nao principal, pois pode mascarar desbalanceamento.

## Metrica de negocio

O custo de negocio considera:

- FP: cliente nao cancelaria, mas recebe acao de retencao.
- FN: cliente cancelaria, mas nao foi identificado.
- Custo de FP: 100.
- Custo de FN: 840.

Formula:

```text
custo_negocio = (FP * 100) + (FN * 840)
```

## Criterio de leitura

Um modelo melhor nao e necessariamente o de maior accuracy. Para este problema, modelos com maior recall/PR AUC e menor custo de negocio tendem a ser mais interessantes, desde que a operacao consiga absorver o volume de clientes acionados.
