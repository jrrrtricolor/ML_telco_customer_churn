# Metricas Tecnicas e de Negocio - Fase 1

## 1. Objetivo

Definir as metricas de validacao para o problema de churn combinando:

- desempenho tecnico de classificacao
- impacto financeiro estimado para acao de retencao

## 2. Metricas tecnicas (modelo)

Metricas obrigatorias para comparacao entre modelos:

- `AUC-ROC`
- `PR-AUC`
- `F1 Score`

Metricas complementares:

- `Accuracy`
- `Precision`
- `Recall`

## 3. Metrica de negocio principal

### Custo de churn evitado

Representa o valor financeiro bruto associado a clientes com churn potencialmente evitado por campanha de retencao.

Formula usada no projeto:

`custo_churn_evitado = churn_evitado_estimado * valor_medio_churn_evitado`

Onde:

- `churn_evitado_estimado = TP * taxa_sucesso_retencao`
- `TP` = clientes corretamente classificados como churn

## 4. Metrica de negocio derivada

### Retorno liquido estimado

Formula:

`retorno_liquido_estimado = custo_churn_evitado - custo_campanha_retencao`

Com:

- `custo_campanha_retencao = contatos_campanha * custo_contato_retencao`
- `contatos_campanha = TP + FP`

## 5. Parametros financeiros default usados na Fase 1

- `valor_medio_churn_evitado = 1000.0`
- `custo_contato_retencao = 40.0`
- `taxa_sucesso_retencao = 0.35`

Esses valores sao premissas iniciais para benchmarking e devem ser recalibrados com dados reais de negocio.

## 6. Como interpretar

- modelo com melhor `AUC-ROC` e `PR-AUC` tende a separar melhor classes
- modelo com melhor `F1` tende a equilibrar precision e recall
- modelo com maior `retorno_liquido_estimado` tende a ser mais util para campanha

Decisao final de deploy deve combinar metrica tecnica e metrica financeira.