# 📡 Previsão de Churn de Clientes — Telco Customer Churn

> Modelo preditivo de aprendizado de máquina para identificar clientes com alto risco de cancelamento em uma operadora de telecomunicações.

---

## 📋 Descrição do Problema

Uma operadora de telecomunicações está perdendo clientes em ritmo acelerado. O time de negócio precisa de um **modelo preditivo de churn** capaz de classificar quais clientes têm maior probabilidade de cancelar o contrato nos próximos meses.

Com essa previsão em mãos, a empresa poderá agir proativamente — oferecendo benefícios, descontos ou melhorias de serviço — antes que o cliente efetivamente cancele.

---

## 🎯 Objetivo do Projeto

- Construir um pipeline completo de Machine Learning para previsão de churn
- Realizar análise exploratória dos dados (EDA) com visualizações detalhadas
- Engenharia de features para maximizar o poder preditivo do modelo
- Treinar, avaliar e comparar múltiplos algoritmos de classificação
- Garantir reprodutibilidade e boas práticas de engenharia de software

---

## 🏗️ Arquitetura do Projeto

```
ML_telco_customer_churn/
│
├── src/                         # Código-fonte principal do projeto
│   ├── data/                    # Scripts de carga e pré-processamento de dados
│   ├── features/                # Engenharia de features
│   ├── models/                  # Treinamento e inferência dos modelos
│   ├── evaluation/              # Métricas e avaliação dos modelos
│   ├── servico/                 # Camada de serviço/inferência
│   └── utils/                   # Funções utilitárias compartilhadas
│
├── data/
│   ├── raw/                     # Dados brutos originais (não modificados)
│   └── processed/               # Dados processados e prontos para modelagem
│
├── models/
│   └── trained_models/          # Modelos treinados serializados (.pkl, .joblib)
│
├── notebooks/                   # Jupyter Notebooks de exploração e experimentos
│
├── tests/                       # Testes automatizados
│
├── docs/                        # Documentação adicional do projeto
├── docker/                      # Arquivos de containerização
├── deployment/                  # Artefatos de deploy
│
├── pyproject.toml               # Configuração do projeto e dependências
├── .gitignore                   # Arquivos e pastas ignorados pelo Git
└── README.md                    # Este arquivo
```

---

## 🛠️ Tecnologias Utilizadas

| Categoria          | Tecnologia                                    |
|--------------------|-----------------------------------------------|
| Linguagem          | Python >= 3.10                                |
| Manipulação de dados | pandas, numpy                               |
| Machine Learning   | scikit-learn                                  |
| Experiment Tracking| MLflow                                        |
| Visualização       | matplotlib, seaborn                           |
| Notebooks          | Jupyter                                       |
| Variáveis de ambiente | python-dotenv                              |
| Testes             | pytest                                        |
| Formatação         | black                                         |
| Linting            | ruff                                          |

---

## ⚙️ Instalação do Ambiente

### Pré-requisitos

- Python >= 3.10
- [pip](https://pip.pypa.io/en/stable/) ou [uv](https://github.com/astral-sh/uv)

### 1. Clone o repositório

```bash
git clone https://github.com/jrrrtricolor/ML_telco_customer_churn.git
cd ML_telco_customer_churn
```

### 2. Crie e ative um ambiente virtual

```bash
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# ou
.venv\Scripts\activate           # Windows
```

### 3. Instale as dependências

```bash
pip install -e ".[dev]"
```

### 4. Instale o DVC

[Veja aqui](https://doc.dvc.org/start)

- Configure suas credenciais do S3
- Executar `dvc pull`

---

## 📓 Como Executar os Notebooks

```bash
jupyter notebook notebooks/
```

Os notebooks estão organizados na pasta `notebooks/` e devem ser executados na ordem numérica indicada no nome dos arquivos.

---

## 🤖 Como Treinar o Modelo

```bash
python -m src.main
```

O pipeline executa pré-processamento, treino, avaliação e tracking no MLflow.

Execute os comandos na raiz do projeto para manter o tracking consistente.

Os modelos treinados serão salvos em `models/trained_models/`.

Executar interface de experimentos do MLflow:

```bash
mlflow ui --backend-store-uri "sqlite:///$(pwd)/mlflow.db" --port 5000
```

Guia de aprendizado passo a passo (iniciante em MLE):

`docs/guia_iniciante_machine_learning_engineering.md`

---

## ✅ Entregas da Fase 1

As atividades da Fase 1 estao formalizadas nos seguintes documentos:

- `docs/ml_canvas_fase1.md`
- `docs/eda_data_readiness_fase1.md`
- `docs/plano_experimentos.md`
- `docs/metricas_negocio.md`
- `report/relatorio_fase1.md`

Evidencias tecnicas implementadas no codigo:

- baselines com `DummyClassifier`, `DecisionTreeClassifier`, `RandomForestClassifier` e `KNeighborsClassifier`
- metricas tecnicas: `AUC-ROC`, `PR-AUC`, `F1`
- metrica de negocio: custo de churn evitado e retorno liquido estimado
- tracking no MLflow com parametros, metricas, artefato de modelo e hash do dataset

---

## 🧪 Como Rodar os Testes

```bash
pytest tests/ -v
```

Para verificar a cobertura de testes:

```bash
pytest tests/ --cov=src --cov-report=term-missing
```

---

## 🔒 Hook de Commit (flake8)

Para bloquear commits fora do padrão flake8:

```bash
git config core.hooksPath .githooks
chmod +x .githooks/pre-commit
python3 -m pip install -e ".[dev]"
```

---

## 🔬 Boas Práticas de Desenvolvimento

Para manter o histórico Git limpo e o projeto organizado:

- **Commits atômicos**: cada commit representa uma única mudança lógica
- **Mensagens de commit semânticas**: `feat:`, `fix:`, `docs:`, `chore:`, `test:`, `refactor:`
- **Branches por funcionalidade**: use `feature/nome-da-feature` ou `fix/nome-do-fix`
- **Linting e formatação automática** antes de todo commit:
  ```bash
  ruff check src/ tests/
  black src/ tests/
  ```
- **Dados nunca versionados diretamente**: dados brutos e processados ficam fora do Git (via `.gitignore`). Use armazenamento externo (S3, DVC, etc.)
- **Variáveis sensíveis em `.env`**: nunca faça commit de credenciais ou chaves de API

---

## 📝 Sugestão de Commits Iniciais

```
feat: criar estrutura inicial do projeto
docs: adicionar README do projeto
build: configurar dependências no pyproject.toml
chore: adicionar .gitignore para projeto de ML
test: criar estrutura inicial de testes
```

---

## 🚀 Próximos Passos

- [ ] Ampliar cobertura de testes automatizados
- [ ] Implementar camada de servico (FastAPI) para inferencia
- [ ] Containerizar treino e servico com Docker
- [ ] Adicionar pipeline de CI/CD
- [ ] Incluir monitoramento de drift e performance em producao

---

## 👤 Autor

Projeto desenvolvido como estudo de caso de Machine Learning aplicado ao setor de telecomunicações.

---

## 📄 Licença

Este projeto está licenciado sob a [MIT License](LICENSE).
