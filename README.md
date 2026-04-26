# ML Telco Customer Churn

Projeto de Machine Learning para previsão de churn (cancelamento) de clientes de telecomunicações.

Este repositório foi estruturado para o Tech Challenge (FIAP), com pipeline de dados, treinamento de baselines (Scikit-Learn), MLP com PyTorch, rastreamento de experimentos com MLflow e API de inferência com FastAPI.

## Nível atual do projeto

- Estágio: entrega acadêmica end-to-end para o Tech Challenge FIAP.
- O fluxo principal de treino roda localmente.
- A suíte de testes roda pelo `Makefile` ou diretamente com `pytest`.
- A API local/Docker serve o modelo registrado no MLflow.

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
pip install -e ".[dev]"
```

## Como executar

### 1) Rodar pipeline de treino e avaliação

```bash
make train
```

Resultado esperado:
- treino dos modelos
- log de métricas
- registro de execuções e modelos no MLflow local (`mlflow.db`)

### 2) Rodar testes

```bash
make test
```

### 3) Verificar lint (qualidade de código)

```bash
make lint
```

### 4) Subir API local

```bash
make api
```

Teste rápido:

```bash
curl http://127.0.0.1:8000/health
```

### 4.1) Subir API com Docker

```bash
docker build -t churn-api .
docker run --rm -p 8000:8000 churn-api
```

A imagem executa o treino no build para gerar `mlflow.db` e `mlruns/` antes de iniciar a API. Assim, um clone limpo com o CSV em `data/raw/Telco_Customer_Churn.csv` consegue subir a API sem passos manuais adicionais dentro do container.

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
make mlflow-ui
```

Depois, abra no navegador:
- `http://127.0.0.1:5001`

## Documentação

- Model Card (formato Google adaptado): `docs/model_card.md`
- ML Canvas: `docs/0- ml_canvas_fase1.md`
- Definição de métricas: `docs/1- definicao_metricas.md`
- Deploy e monitoramento: `docs/deploy_monitoramento.md`
- Roteiro do vídeo STAR: `docs/roteiro_video_star.md`

## Estrutura do projeto

```text
src/
data/
docs/
notebooks/
tests/
pyproject.toml
requirement.txt
Makefile
```

## Deploy

O escopo atual usa FastAPI local/Docker para servir o modelo. O deploy em nuvem é bônus opcional do Tech Challenge e não será perseguido nesta versão.

Para demonstração da entrega, rode o treino com `make train`, suba a API com `make api` e valide os endpoints `/health`, `/predict` e `/metrics`.

## Troubleshooting

- Erro `ModuleNotFoundError: No module named 'src'`:
  - execute com `PYTHONPATH=.` nos comandos Python/Pytest.
- Porta em uso na API:
  - altere `--port` no comando do uvicorn.
- Ambiente inconsistente:
  - recrie o virtualenv e reinstale dependências.

## Backlog técnico prioritário

1. Atualizar resultados finais no Model Card sempre que o treino for reexecutado.
2. Evoluir análise de fairness por segmento com dados recentes.
3. Avaliar deploy em nuvem apenas como evolução futura.

## Licença

MIT
