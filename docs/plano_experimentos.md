# Plano de Experimentos
## Projeto de Previsão de Churn

---

# 1. Objetivo

Este documento descreve os experimentos que serão realizados durante o desenvolvimento do modelo de Machine Learning.

---

# 2. Dataset

Dataset utilizado:

Telco Customer Churn

Fonte:
https://www.kaggle.com/datasets/blastchar/telco-customer-churn

---

# 3. Etapas do Experimento

## Etapa 1 — Exploração dos Dados

Objetivo:

Entender os dados disponíveis.

Perguntas importantes:

- Qual a taxa de churn do dataset?
- Clientes com contrato mensal cancelam mais?
- Clientes novos cancelam mais?
- Clientes com mensalidade alta cancelam mais?

Ferramentas:

- Python
- Pandas
- Matplotlib
- Seaborn

---

## Etapa 2 — Preparação dos Dados

Atividades:

- tratamento de valores nulos
- conversão de variáveis categóricas
- preparação da variável alvo

Conversão da coluna churn:

Yes → 1  
No → 0

Separação dos dados:

80% treino  
20% teste

---

## Etapa 3 — Modelo Baseline

Treinar um modelo simples para servir como referência.

Modelo inicial sugerido:

Logistic Regression

---

## Etapa 4 — Avaliação do Modelo

Métricas utilizadas:

- Accuracy
- Precision
- Recall
- F1 Score
- AUC

---

## Etapa 5 — Melhorias no Modelo

Possíveis melhorias:

- testar outros algoritmos
- ajustar parâmetros
- selecionar melhores features

Modelos possíveis:

- Decision Tree
- Random Forest
- Gradient Boosting

---

# 4. Registro dos Experimentos

Os experimentos serão documentados nos notebooks.

Estrutura sugerida:

notebooks/

01_exploracao_dados.ipynb  
02_preparacao_dados.ipynb  
03_modelo_baseline.ipynb

---

# 5. Resultado Esperado

Ao final dos experimentos esperamos obter um modelo capaz de prever churn com boa precisão.