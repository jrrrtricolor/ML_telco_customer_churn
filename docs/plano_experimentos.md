# Plano de Experimentos - Fase 1

## 1. Objetivo

Executar a trilha completa da Fase 1 para previsao de churn com foco em:

- baseline estatistico (`DummyClassifier`)
- baselines supervisionados de arvore e vizinhanca (`DecisionTree`, `RandomForest`, `KNN`)
- comparacao entre modelos classicos de classificacao
- registro de experimentos no MLflow com parametros, metricas e versionamento do dataset

## 2. Dataset e versionamento

- Fonte: Telco Customer Churn (Kaggle)
- Arquivo: `data/raw/Telco_Customer_Churn.csv`
- Versionamento no tracking: `SHA-256` do arquivo bruto logado por run no MLflow (`dataset_sha256`)

## 3. Etapas de execucao

### Etapa 1 - EDA e data readiness

- volume, qualidade, distribuicao e riscos
- verificacao de consistencia de tipos
- mapeamento de problemas de dados (`TotalCharges` com valores em branco)

### Etapa 2 - Preparacao de dados

- remocao de coluna de identificador (`customerID`)
- encoding da variavel alvo (`Yes -> 1`, `No -> 0`)
- split estratificado treino/teste: `80/20`
- preprocessamento em `Pipeline` para evitar leakage

### Etapa 3 - Baselines e modelos comparativos

Modelos treinados:

- `DummyClassifier(strategy="most_frequent")`
- `DecisionTreeClassifier`
- `RandomForestClassifier`
- `KNeighborsClassifier`

### Etapa 4 - Avaliacao tecnica

Metricas por modelo:

- `accuracy`
- `precision`
- `recall`
- `f1`
- `roc_auc`
- `pr_auc`

### Etapa 5 - Avaliacao de negocio

Metricas de impacto para retencao:

- `contatos_campanha`
- `churn_evitado_estimado`
- `custo_churn_evitado`
- `custo_campanha_retencao`
- `retorno_liquido_estimado`

## 4. Tracking no MLflow

Cada run registra:

- parametros globais (`test_size`, `random_state`)
- parametros do estimador
- metricas tecnicas e de negocio
- assinatura/versionamento do dataset (`dataset_sha256`)
- artefato do modelo (`mlflow.sklearn.log_model`)
- artefato auxiliar `dataset_version.json`

## 5. Comandos da Fase 1

Treino e tracking:

```bash
python -m src.main
```

Abrir UI do MLflow local:

```bash
mlflow ui --backend-store-uri "sqlite:///$(pwd)/mlflow.db" --port 5000
```

## 6. Critérios de aceite da Fase 1

- ML Canvas preenchido com stakeholders, metricas de negocio e SLOs
- EDA registrada com volume, qualidade, distribuicao e readiness
- baselines e modelos classicos treinados e comparados
- runs disponiveis no MLflow com parametros, metricas e dataset version