# Model Card (Google Adaptado) - Predição de Churn (Telco)

## 1. Model Details (Detalhes do Modelo)

- **Nome do projeto**: `ML Telco Customer Churn`
- **Tipo de problema**: classificação binária (`0 = não churn`, `1 = churn`)
- **Tecnologias**: Scikit-Learn, PyTorch, MLflow, FastAPI
- **Registro de artefatos**: MLflow local (`mlflow.db`)
- **Data de referência**: 2026-04-13
- **Maturidade atual**: acadêmico (iniciante/intermediário)
- **Autores**: equipe do Tech Challenge FIAP

## 2. Intended Use (Uso Pretendido)

### 2.1 Primary Intended Uses

- Priorizar clientes com maior risco de cancelamento para campanhas de retenção.
- Apoiar analistas de CRM e negócio na tomada de decisão.
- Uso recomendado em processamento batch (listas por ciclo).

### 2.2 Out-of-Scope Uses

- Decisão totalmente automatizada sem supervisão humana.
- Uso em domínios regulados (crédito, saúde, justiça, segurança) sem validação específica.
- Uso em tempo real com SLA rigoroso sem evolução da arquitetura atual.

## 3. Factors (Fatores Relevantes)

Fatores que podem impactar desempenho e risco:

- Mudança de distribuição do perfil dos clientes ao longo do tempo (drift).
- Diferenças entre segmentos (ex.: senioridade, contrato, forma de pagamento).
- Qualidade e completude das variáveis de entrada.
- Regra de imputação de `TotalCharges`, que pode afetar alguns subgrupos.

## 4. Metrics (Métricas)

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

## 5. Evaluation Data (Dados de Avaliação)

- Dataset: `data/raw/Telco_Customer_Churn.csv`
- Estratégia atual: divisão treino/teste + validação cruzada estratificada no treino
- Unidade de análise: 1 linha por cliente
- Target: `Churn` mapeado de `Yes/No` para `1/0`

### 5.1 Resultados de Referência (snapshot)

| Modelo | Accuracy | Precision | Recall | F1 | ROC AUC | PR AUC | Custo de negócio |
|---|---:|---:|---:|---:|---:|---:|---:|
| MLP | 0.7317 | 0.5143 | 0.7722 | 0.6174 | 0.7945 | 0.5359 | 104400 |
| Decision Tree | 0.7204 | 0.5013 | 0.4785 | 0.4896 | 0.6476 | 0.3866 | 191840 |
| Dummy | 0.7197 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.2803 | 331800 |
| KNN | 0.7701 | 0.6060 | 0.5139 | 0.5562 | 0.8048 | 0.5603 | 174480 |
| Random Forest | 0.7842 | 0.6502 | 0.4987 | 0.5645 | 0.8264 | 0.6385 | 176920 |
| Logistic Regression | a atualizar | a atualizar | a atualizar | a atualizar | a atualizar | a atualizar | a atualizar |

Leitura rápida:

- Melhor custo de negócio no snapshot: `MLP`.
- Melhores AUCs no snapshot: `Random Forest` e `KNN`.

## 6. Training Data (Dados de Treinamento)

Pré-processamento aplicado no estado atual:

- Remoção de `customerID`.
- Conversão numérica de `TotalCharges`, `MonthlyCharges` e `tenure`.
- Remoção de registros com `MonthlyCharges <= 0`.
- Imputação de `TotalCharges` com `tenure * MonthlyCharges`.
- Encoding de categóricas com transformador custom.
- Imputação mediana + padronização no pipeline sklearn.

Limitações conhecidas de treinamento:

- A validação cruzada estratificada já está no fluxo principal, mas ainda sem busca de hiperparâmetros.
- A baseline de Regressão Logística foi adicionada para comparação formal.
- MLP com validação, mini-batch e early stopping em versão inicial.

## 7. Ethical Considerations (Considerações Éticas)

Riscos potenciais:

- Variáveis sociodemográficas podem gerar disparidades entre grupos.
- O histórico pode refletir vieses operacionais anteriores.
- Threshold único pode impactar segmentos de forma desigual.

Boas práticas recomendadas:

- Avaliar métricas por segmento antes de uso em produção.
- Comparar taxas de FP/FN por grupo.
- Manter supervisão humana na decisão final de negócio.

## 8. Caveats and Recommendations (Ressalvas e Recomendações)

### 8.1 Caveats

- Projeto em nível acadêmico, ainda não pronto para produção enterprise.
- A suíte de testes depende de `PYTHONPATH=.` no estado atual.
- O lint (`ruff`) ainda apresenta pendências.
- A API carrega modelo por `latest`, reduzindo o controle fino de versão.

### 8.2 Recommendations (Próximos Passos)

1. Evoluir tuning de hiperparâmetros (baseline e MLP).
2. Adicionar testes de schema (`pandera`) e testes de API/smoke.
3. Corrigir pendências de lint e padronizar naming/type hints.
4. Definir monitoramento contínuo (drift, performance e custo de negócio).

## 9. Como Citar este Model Card

Documento interno do projeto: `docs/model_card.md`.
Uso principal: entrega e auditoria técnica do Tech Challenge FIAP.
