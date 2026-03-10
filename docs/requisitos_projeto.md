# Documento de Requisitos do Projeto
## Previsão de Churn de Clientes

---

# 1. Contexto do Problema

Uma operadora de telecomunicações está perdendo clientes em ritmo acelerado.  
Quando um cliente cancela o serviço, isso é chamado de **churn**.

A perda de clientes impacta diretamente a receita da empresa e aumenta os custos para adquirir novos clientes.

Por esse motivo, a empresa deseja utilizar **Machine Learning** para prever quais clientes possuem maior risco de cancelamento.

---

# 2. Objetivo do Projeto

O objetivo deste projeto é desenvolver um modelo de Machine Learning capaz de prever quais clientes possuem maior probabilidade de cancelar o serviço.

Com essa previsão, a empresa poderá:

- identificar clientes com risco de churn
- criar campanhas de retenção
- reduzir perdas financeiras

---

# 3. Definição de Churn

Neste projeto, a variável alvo será a coluna:

Churn

Valores possíveis:

Yes → cliente cancelou  
No → cliente continua ativo

Para o modelo de Machine Learning:

Yes = 1  
No = 0

---

# 4. Base de Dados

Dataset utilizado:

Telco Customer Churn

Disponível em:
https://www.kaggle.com/datasets/blastchar/telco-customer-churn

Cada linha do dataset representa um cliente.

---

# 5. Features Utilizadas

As seguintes colunas serão utilizadas no modelo.

## Dados do cliente

- gender
- SeniorCitizen
- Partner
- Dependents

## Informações do contrato

- tenure
- Contract
- PaperlessBilling
- PaymentMethod

## Serviços utilizados

- PhoneService
- InternetService
- OnlineSecurity
- TechSupport
- StreamingTV
- StreamingMovies

## Informações financeiras

- MonthlyCharges
- TotalCharges

## Variável alvo

- Churn

---

# 6. Estrutura do Modelo (Árvore de Decisão)

Para facilitar o entendimento, será utilizada uma árvore de decisão simples com no máximo **5 nós**.

Cada nó representa uma pergunta sobre o cliente.

### Nó 1

Pergunta:

O cliente possui contrato mensal?

Feature utilizada:
Contract

---

### Nó 2

Pergunta:

O cliente está na empresa há menos de 12 meses?

Feature utilizada:
tenure

---

### Nó 3

Pergunta:

O valor da mensalidade do cliente é maior que a média?

Feature utilizada:
MonthlyCharges

---

### Nó 4

Pergunta:

O cliente possui suporte técnico?

Feature utilizada:
TechSupport

---

### Nó 5

Pergunta:

O cliente possui serviço de segurança online?

Feature utilizada:
OnlineSecurity

---

# 7. Métricas de Sucesso

Para avaliar o modelo serão utilizadas as seguintes métricas:

Accuracy > 70%

Recall ≥ 70%

AUC ≥ 0.75

---

# 8. Resultado Esperado

Ao final do projeto espera-se obter um modelo capaz de prever churn de clientes e ajudar a empresa a reduzir cancelamentos.