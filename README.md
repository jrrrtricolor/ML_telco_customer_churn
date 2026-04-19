# ML Telco Customer Churn

Projeto de Machine Learning para previsão de churn (cancelamento) de clientes de telecom.

Este repositório foi estruturado para o Tech Challenge (FIAP), com pipeline de dados, treinamento de modelos baseline (Scikit-Learn), MLP com PyTorch, rastreamento no MLflow e API com FastAPI.

## Nível atual do projeto

- Estágio: **iniciante para intermediário**.
- O fluxo principal de treino roda localmente.
- A suíte de testes roda localmente quando `PYTHONPATH=.` é usado.
- Ainda há melhorias importantes para nível de produção (documentadas na seção de backlog).

## Arquitetura (visão geral)

- `src/load.py`: carga e validação inicial dos dados.
- `src/data_prep.py`: limpeza, conversão de colunas e split treino/teste.
- `src/sklearn_pipeline.py` + `src/data_classifier.py`: pré-processamento e pipeline sklearn.
- `src/model_factory.py`: fábrica de modelos baseline + MLP.
- `src/sklearn_mlp_model.py` + `src/mlp_model.py`: implementação de MLP em PyTorch com interface sklearn.
- `src/trainer.py`: treino e inferência dos modelos.
- `src/avaliador.py`: métricas técnicas e custo de negócio.
- `src/pipeline.py`: orquestração fim a fim + logs no MLflow.
- `src/api.py`: API FastAPI (`/health`, `/predict`).
- `tests/`: testes unitários.

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

1. Remover necessidade de `PYTHONPATH=.` com empacotamento/instalação correta do módulo `src`.
2. Expandir testes (API, schema e smoke test).
3. Evoluir o Model Card com análises por segmento (fairness) e histórico de versões.

## Licença

MIT
