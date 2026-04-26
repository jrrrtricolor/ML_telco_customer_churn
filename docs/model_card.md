# Model Card (Google Adaptado) - Predição de Churn (Telco)

## 1. Detalhes do Modelo

- **Nome do projeto**: `ML Telco Customer Churn`
- **Tipo de problema**: classificação binária (`0 = não churn`, `1 = churn`)
- **Tecnologias**: Scikit-Learn, PyTorch, MLflow, FastAPI
- **Registro de artefatos**: MLflow local (`mlflow.db`)
- **Data de referência**: 2026-04-26
- **Maturidade atual**: acadêmico (iniciante/intermediário)
- **Autores**: equipe do Tech Challenge FIAP

## 2. Uso Pretendido

### 2.1 Usos principais

- Priorizar clientes com maior risco de cancelamento para campanhas de retenção.
- Apoiar analistas de CRM e negócio na tomada de decisão.
- Uso recomendado em processamento em lote (listas por ciclo).

### 2.2 Usos fora de escopo

- Decisão totalmente automatizada sem supervisão humana.
- Uso em domínios regulados (crédito, saúde, justiça, segurança) sem validação específica.
- Uso em tempo real com SLA rigoroso sem evolução da arquitetura atual.

## 3. Fatores Relevantes

Fatores que podem impactar desempenho e risco:

- Mudança de distribuição do perfil dos clientes ao longo do tempo (drift).
- Diferenças entre segmentos (ex.: senioridade, contrato, forma de pagamento).
- Qualidade e completude das variáveis de entrada.
- Regra de imputação de `TotalCharges`, que pode afetar alguns subgrupos.

## 4. Métricas

Métricas técnicas monitoradas:

- Accuracy
- Precision
- Recall
- F1-score
- ROC AUC
- PR AUC

Métrica de negócio:

- `custo_negocio = (FP * 100) + (FN * 840)`

> Observação: no contexto de churn, falsos negativos (FN) tendem a gerar maior impacto financeiro.

## 5. Dados de Avaliação

- Dataset: `data/raw/Telco_Customer_Churn.csv`
- Estratégia atual: divisão treino/teste estratificada + validação cruzada estratificada no treino
- Unidade de análise: 1 linha por cliente
- Target: `Churn` mapeado de `Yes/No` para `1/0`

### 5.1 Resultados de referência (snapshot)

| Modelo | Accuracy | Precision | Recall | F1 | ROC AUC | PR AUC | Custo de negócio |
|---|---:|---:|---:|---:|---:|---:|---:|
| MLP | 0.7942 | 0.6391 | 0.5160 | 0.5710 | 0.8271 | 0.6371 | 162940 |
| Decision Tree | 0.7480 | 0.5266 | 0.5027 | 0.5144 | 0.6701 | 0.3974 | 173140 |
| Dummy | 0.7346 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.2654 | 314160 |
| KNN | 0.7601 | 0.5542 | 0.4920 | 0.5212 | 0.7644 | 0.5020 | 174400 |
| Random Forest | 0.7814 | 0.6146 | 0.4733 | 0.5347 | 0.8134 | 0.5947 | 176580 |
| Logistic Regression | 0.8013 | 0.6506 | 0.5428 | 0.5918 | 0.8304 | 0.6445 | 154540 |

Leitura rápida:

- Melhor custo de negócio no snapshot: `Logistic Regression`.
- Melhor PR-AUC no snapshot: `Logistic Regression`.
- A MLP continua como entrega central do desafio, mas ainda precisa de tuning para superar o baseline linear.

## 6. Dados de Treinamento

Pré-processamento aplicado no estado atual:

- Remoção de `customerID`.
- Conversão numérica de `TotalCharges`, `MonthlyCharges` e `tenure`.
- Remoção de registros com `MonthlyCharges <= 0`.
- Imputação de `TotalCharges` com `tenure * MonthlyCharges`.
- Encoding de categóricas com transformador custom.
- Imputação mediana + padronização no pipeline sklearn.

Limitações conhecidas de treinamento:

- A validação cruzada estratificada está no fluxo principal, mas ainda sem busca de hiperparâmetros.
- A baseline de Regressão Logística foi adicionada para comparação formal.
- MLP com validação, mini-batch e early stopping em versão inicial.

## 7. Considerações Éticas

Riscos potenciais:

- Variáveis sociodemográficas podem gerar disparidades entre grupos.
- O histórico pode refletir vieses operacionais anteriores.
- Threshold único pode impactar segmentos de forma desigual.

Boas práticas recomendadas:

- Avaliar métricas por segmento antes de uso em produção.
- Comparar taxas de FP/FN por grupo.
- Manter supervisão humana na decisão final de negócio.

## 8. Ressalvas e Recomendações

### 8.1 Ressalvas

- Projeto em nível acadêmico, ainda não pronto para produção enterprise.
- A suíte de testes rápidos passa localmente sem depender do Docker.
- O lint (`ruff`) está sem erros no estado atual.
- A API carrega modelo por `latest` quando `MODEL_URI` não é informado, reduzindo o controle fino de versão.

### 8.2 Recomendações (próximos passos)

1. Evoluir o tuning de hiperparâmetros da MLP.
2. Rodar teste de schema com `pandera` instalado no ambiente.
3. Executar e2e Docker em ambiente com Docker disponível.
4. Definir monitoramento contínuo em nuvem (drift, performance e custo de negócio).

## 9. Como citar este Model Card

Documento interno do projeto: `docs/model_card.md`.
Uso principal: entrega e auditoria técnica do Tech Challenge FIAP.
