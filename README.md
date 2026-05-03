# ML Telco Customer Churn

Projeto de Machine Learning para previsão de churn (cancelamento) de clientes de telecomunicações.

Este repositório foi estruturado para o Tech Challenge (FIAP), com pipeline de dados, treinamento de baselines (Scikit-Learn), MLP com PyTorch, rastreamento de experimentos com MLflow e API de inferência com FastAPI.

## Nível atual do projeto

- Estágio: entrega acadêmica end-to-end para o Tech Challenge FIAP.
- O fluxo principal de treino roda localmente.
- A suíte de testes roda pelo `Makefile` ou diretamente com `pytest`.
- A API local/Docker serve o modelo registrado no MLflow.

## Acesso para avaliação FIAP

- Repositório GitHub: https://github.com/jrrrtricolor/ML_telco_customer_churn
- Branch principal para avaliação: `main`
- Vídeo de apresentação: https://youtu.be/Ib49IrSbYhc

### Checklist dos requisitos atendidos

| Requisito | Onde validar |
|---|---|
| Problema de negócio e proposta de valor | `README.md`, `docs/0- ml_canvas_fase1.md` |
| Pipeline de preparação e treino | `src/data_prep.py`, `src/pipeline.py`, `src/main.py` |
| Baselines Scikit-Learn | `src/model_factory.py`, `src/sklearn_pipeline.py` |
| Rede neural MLP com PyTorch | `src/mlp_model.py`, `src/sklearn_mlp_model.py` |
| Métricas técnicas e métrica de negócio | `src/avaliador.py`, `docs/1- definicao_metricas.md` |
| Rastreamento de experimentos com MLflow | `src/pipeline.py`, comando `make mlflow-ui` |
| API de inferência com FastAPI | `src/api.py`, endpoints `/health`, `/predict`, `/metrics` |
| Validação de entrada | `src/schema.py`, testes em `tests/test_schema.py` |
| Docker para execução reprodutível | `Dockerfile`, comando `docker build -t churn-api .` |
| Testes automatizados | `tests/`, comando `make test` |
| Lint/qualidade de código | `pyproject.toml`, comando `make lint` |
| Monitoramento básico | `src/prometheus/`, `docs/deploy_monitoramento.md` |
| Documentação de riscos e limitações | `docs/model_card.md`, `docs/deploy_monitoramento.md` |

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

## Dados

O dataset principal esperado pelo pipeline está em:

```text
data/raw/Telco_Customer_Churn.csv
```

O repositório também mantém arquivos `.dvc` para rastreabilidade dos dados. Em um clone limpo da entrega, valide se o CSV está presente antes de rodar `make train` ou o build Docker.

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

## Validação da entrega

Comandos recomendados antes da avaliação:

```bash
git checkout main
git pull origin main
make lint
make test-unit
docker build -t churn-api .
docker run --rm -p 8000:8000 churn-api
```

Em outra aba, valide:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/metrics
```

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

## Video Apresentação

https://youtu.be/Ib49IrSbYhc
