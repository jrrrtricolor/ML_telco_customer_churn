# PROMPT MESTRE - DOCUMENTACAO DE ANALISE TECNICA (CHURN TELECOM)

Use o texto abaixo em um modelo de IA para gerar uma documentacao tecnica completa do projeto.

```text
Voce e um(a) Cientista de Dados Senior com foco em ML aplicado a telecom e MLOps.

Sua tarefa e gerar uma documentacao de analise tecnica completa para um projeto de Ciencia de Dados que preve churn de clientes de uma central de telecomunicacoes.

## Requisitos de projeto
- Deve ser entregue uma API Http para inferência, aceitando apenas 1 registro por chamada
- Deve ser utilizada uma abordagem online
- Não utilizar abordagem em batch

## Objetivo da documentacao
Produzir um documento tecnico em Markdown que permita a qualquer pessoa do time (dados, negocio, engenharia e lideranca) entender:
1) Qual o problema a ser tratado
2) Quais métricas definirão o sucesso do projeto (KPIs e métricas técnicas)
3) Qual o prazo e orçamento para o projeto
4) Quais informações temos disponíveis
5) Quais dados são protegidos pela LGPD
6) Quais são os departamentos de Stakeholders que devem ser envolvidos.
7) Quais pessoas estarão alocadas para desenvolver o projeto

## Contexto do projeto
- Dominio: telecomunicacoes.
- Problema: prever probabilidade de cancelamento de clientes (churn).
- Tipo de problema: classificacao binaria supervisionada.
- Variavel alvo esperada: churn (Yes/No ou 1/0).
- Objetivo de negocio: reduzir cancelamentos com acoes preventivas orientadas por score de risco.

### Colunas disponíveis na base de dados: 

#### Dados do cliente
- customerID
- gender
- SeniorCitizen
- Partner
- Dependents
- tenure

#### Serviços contratados
- PhoneService
- MultipleLines
- InternetService
- OnlineSecurity
- OnlineBackup
- DeviceProtection
- TechSupport
- StreamingTV
- StreamingMovies

#### Informações contratuais
- Contract
- PaperlessBilling
- PaymentMethod
- MonthlyCharges
- TotalCharges
- Churn

## Entradas a serem geradas conforme padrões existentes no mercado
- Métricas de sucesso técnico
- Métricas de sucesso (KPIs)
- Orçamento para o projeto
- Possíveis Stakeholders

Analisar as informações disponíveis quanto a restrições conforme a lei LGPD

## Estrutura obrigatoria da saida (Markdown)
Gere o documento com estas secoes, nesta ordem:

1. Resumo tecnico executivo
- Problema

2. Escopo e criterio de sucesso
- Pergunta analitica principal.
- Definicao de sucesso de negocio (KPI de retencao, custo evitado, uplift esperado).
- Definicao de sucesso tecnico (metricas alvo) (acurácia, precisão, recall, F1, latência de API, disponibilidade de API, taxa de Erro de API).

3. Dados disponíveis e LGPD

4. Lista de stakeholders

5. Lista de engenheiros

4. Riscos

## Entradas disponiveis 
- Prazo para o projeto: 05/05/2026
- Pessoas que serão alocadas no projeto como engenherios de ML: Cássio, Júlia, Lara e Rayane

## Regras de qualidade da resposta
- Escreva em portugues do Brasil, com linguagem tecnica clara e objetiva.
- Use tom profissional, sem marketing e sem exageros.
- Seja especifico e auditavel: toda afirmacao deve ter base tecnica.
- Não Inclua formulas das metricas
- Nao use texto generico; contextualize para churn em telecom.

## Formato final esperado
- Entregar apenas o documento tecnico final em Markdown.
- Nao incluir explicacoes sobre como voce pensou.
- Nao incluir blocos de "prompt" na saida.
- Não incluir secoes não solicitadas
- Não mencionar papéis para cada um dos engenheiros de ML
- Não analisar os dados. Apenas trabalhar com requisitos de negócio
```
