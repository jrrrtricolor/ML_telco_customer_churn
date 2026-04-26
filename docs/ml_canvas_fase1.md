# ML Canvas - Fase 1 (Churn Telco)

## 1. Problema de negocio

A operadora precisa reduzir cancelamentos de clientes (churn) para proteger receita recorrente e diminuir CAC de reposicao.

## 2. Objetivo de negocio

- identificar clientes com maior risco de churn para acionar retencao proativa
- reduzir perda de receita por cancelamento
- priorizar acoes de relacionamento com base em propensao

## 3. Stakeholders

- patrocinador executivo: diretoria de receita/comercial
- dono do produto: time de CRM/retencao
- usuarios de decisao: analistas de marketing de relacionamento
- time tecnico: ciencia de dados + engenharia de ML
- risco e compliance: governanca de dados

## 4. Decisao suportada pelo modelo

Para cada cliente, gerar score/probabilidade de churn e classe prevista para:

- ranquear carteira por risco
- selecionar clientes para campanha de retencao
- definir priorizacao por retorno financeiro esperado

## 5. Metricas tecnicas

- principal: AUC-ROC, PR-AUC, F1
- secundarias: Accuracy, Precision, Recall

## 6. Metrica de negocio

Metrica principal de negocio: custo de churn evitado.

Formula:

`custo_churn_evitado = TP * taxa_sucesso_retencao * valor_medio_churn_evitado`

Metrica derivada:

`retorno_liquido_estimado = custo_churn_evitado - (TP + FP) * custo_contato_retencao`

## 7. SLOs da Fase 1

- qualidade preditiva minima:
  - AUC-ROC >= 0.75
  - Recall classe churn >= 0.70
- confiabilidade de execucao:
  - pipeline de treino concluido sem falha em >= 99% das execucoes
- desempenho operacional:
  - treino completo em ate 10 minutos no ambiente local padrao

## 8. Restricoes e premissas

- dados historicos em arquivo CSV local
- labels binarios (Yes/No) para churn
- campanha de retencao modelada com premissas financeiras default na Fase 1

## 9. Fora de escopo da Fase 1

- deploy em producao da API
- monitoramento online de drift e performance em tempo real
- automacao CI/CD completa
