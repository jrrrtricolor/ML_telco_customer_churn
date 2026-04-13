# ML Telco Customer Churn

Projeto de Machine Learning para previsao de churn (cancelamento) de clientes de telecom.

Este repositorio foi estruturado para o Tech Challenge (FIAP), com pipeline de dados, treinamento de modelos baseline (Scikit-Learn), MLP com PyTorch, rastreamento no MLflow e API com FastAPI.

## Nivel atual do projeto

- Estagio: **iniciante para intermediario**.
- O fluxo principal de treino roda localmente.
- A suite de testes roda localmente quando `PYTHONPATH=.` e usado.
- Ainda ha melhorias importantes para nivel de producao (documentadas na secao de backlog).

## Arquitetura (visao geral)

- `src/load.py`: carga e validacao inicial dos dados.
- `src/data_prep.py`: limpeza, conversao de colunas e split treino/teste.
- `src/sklearn_pipeline.py` + `src/data_classifier.py`: pre-processamento e pipeline sklearn.
- `src/model_factory.py`: fabrica de modelos baseline + MLP.
- `src/sklearn_mlp_model.py` + `src/mlp_model.py`: implementacao de MLP em PyTorch com interface sklearn.
- `src/trainer.py`: treino e inferencia dos modelos.
- `src/avaliador.py`: metricas tecnicas e custo de negocio.
- `src/pipeline.py`: orquestracao fim a fim + logs no MLflow.
- `src/api.py`: API FastAPI (`/health`, `/predict`).
- `tests/`: testes unitarios.

## Requisitos

- Python 3.11+
- Ambiente virtual (recomendado)
- Dependencias do projeto

## Setup rapido

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirement.txt
```

## Como executar

### 1) Rodar pipeline de treino e avaliacao

```bash
PYTHONPATH=. python -m src.main
```

Resultado esperado:
- treino dos modelos
- log de metricas
- registro de execucoes e modelos no MLflow local (`mlflow.db`)

### 2) Rodar testes

```bash
PYTHONPATH=. pytest -q
```

### 3) Verificar lint (qualidade de codigo)

```bash
ruff check src tests
```

> Observacao: neste momento, o lint ainda aponta pendencias que fazem parte do backlog tecnico.

### 4) Subir API local

```bash
PYTHONPATH=. uvicorn src.api:app --host 127.0.0.1 --port 8000
```

Teste rapido:

```bash
curl http://127.0.0.1:8000/health
```

### 5) Exemplo de inferencia (`/predict`)

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "DSL",
    "OnlineSecurity": "Yes",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 55.2,
    "TotalCharges": 662.4
  }'
```

## MLflow

Para visualizar experimentos localmente:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
```

Depois, abra no navegador:
- `http://127.0.0.1:5001`

## Documentacao

- Model Card (formato Google adaptado): `docs/model_card.md`
- ML Canvas: `docs/0- ml_canvas_fase1.md`
- Definicao de metricas: `docs/1- definicao_metricas.md`

## Estrutura do projeto

```text
src/
data/
docs/
notebooks/
tests/
pyproject.toml
requirement.txt
```

## Troubleshooting

- Erro `ModuleNotFoundError: No module named 'src'`:
  - execute com `PYTHONPATH=.` nos comandos Python/Pytest.
- Porta em uso na API:
  - altere `--port` no comando do uvicorn.
- Ambiente inconsistente:
  - recrie o virtualenv e reinstale dependencias.

## Backlog tecnico prioritario

1. Corrigir todos os erros de lint do `ruff`.
2. Remover necessidade de `PYTHONPATH=.` com empacotamento/instalacao correta do modulo `src`.
3. Incluir baseline de Regressao Logistica para comparacao formal.
4. Evoluir treinamento MLP com validacao, batching e early stopping.
5. Expandir testes (API, schema e smoke test).
6. Evoluir o Model Card com analises por segmento (fairness) e historico de versoes.

## Licenca

MIT
