# ML Telco Customer Churn

Projeto de Machine Learning para previsão de churn (cancelamento) de clientes de telecomunicações.

Este repositório foi estruturado para o Tech Challenge (FIAP), com pipeline de dados, treinamento de baselines (Scikit-Learn), MLP com PyTorch, rastreamento de experimentos com MLflow e API de inferência com FastAPI.

## Nível atual do projeto

- Estágio: **iniciante para intermediário**.
- O fluxo principal de treino roda localmente.
- A suíte de testes roda localmente com `PYTHONPATH=.`.
- Ainda há melhorias importantes para nível de produção (listadas no backlog).

## Arquitetura (visão geral)

- `src/load.py`: carga e validação inicial dos dados.
- `src/data_prep.py`: limpeza, conversão de colunas, split estratificado treino/teste.
- `src/sklearn_pipeline.py` + `src/data_classifier.py`: pré-processamento e pipeline sklearn.
- `src/model_factory.py`: fábrica de baselines + MLP.
- `src/sklearn_mlp_model.py` + `src/mlp_model.py`: MLP em PyTorch com interface sklearn.
- `src/trainer.py`: treino e inferência dos modelos.
- `src/avaliador.py`: métricas técnicas e custo de negócio.
- `src/pipeline.py`: orquestração fim a fim + validação cruzada estratificada + logs no MLflow.
- `src/api.py`: API FastAPI (`/health`, `/predict`).
- `tests/`: testes unitários e teste e2e da API em Docker.

## Requisitos

- Python 3.11+
- Ambiente virtual (recomendado)
- Dependências do projeto

## Setup rápido

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirement.txt
```

## Como executar

### 1) Rodar pipeline de treino e avaliação

```bash
PYTHONPATH=. python -m src.main
```

Resultado esperado:
- treino dos modelos
- log de métricas
- registro de execuções e modelos no MLflow local (`mlflow.db`)

### 2) Rodar testes

```bash
PYTHONPATH=. pytest -q
```

### 3) Verificar lint (qualidade de código)

```bash
ruff check src tests
```

### 4) Subir API local

```bash
PYTHONPATH=. uvicorn src.api:app --host 127.0.0.1 --port 8000
```

Teste rápido:

```bash
curl http://127.0.0.1:8000/health
```

### 5) Exemplo de inferência (`/predict`)

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

## Documentação

- Model Card (formato Google adaptado): `docs/model_card.md`
- ML Canvas: `docs/0- ml_canvas_fase1.md`
- Definição de métricas: `docs/1- definicao_metricas.md`

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
  - recrie o virtualenv e reinstale dependências.

## Backlog técnico prioritário

1. Corrigir os erros de lint do `ruff`.
2. Remover necessidade de `PYTHONPATH=.` com empacotamento/instalação correta do módulo `src`.
3. Expandir testes de schema (`pandera`) e smoke test da API.
4. Evoluir Model Card com análises por segmento (fairness) e histórico de versões.
5. Definir estratégia de deploy e monitoramento em nuvem (Azure).

## Licença

MIT
