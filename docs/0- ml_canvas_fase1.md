# ML Canvas - Previsao de Churn

## Proposta de valor

- Problema: clientes de telecomunicacoes cancelam contratos e reduzem receita recorrente.
- Objetivo: priorizar clientes com maior risco de churn para campanhas de retencao.
- Usuarios: CRM, Retencao, Atendimento e liderancas de negocio.
- Valor esperado: reduzir perdas e direcionar melhor o custo de campanha.

## Dados

- Fonte: `data/raw/Telco_Customer_Churn.csv`.
- Unidade de analise: um cliente por linha.
- Target: `Churn` (`Yes`/`No`), tratado como classificacao binaria.
- Principais grupos de variaveis: perfil, contrato, servicos e cobranca.

## Predicao

- Entrada: atributos cadastrais, contratuais e financeiros do cliente.
- Saida: classe/probabilidade de churn.
- Uso inicial: ranking de risco para acao de retencao.

## Metricas

- Tecnicas: PR AUC, ROC AUC, F1, recall, precision e accuracy.
- Negocio: custo estimado por falso positivo e falso negativo.
- Prioridade: reduzir falsos negativos, pois clientes que cancelam sem acao geram maior perda.

## Requisitos operacionais

- Pipeline reprodutivel com seeds fixos.
- Registro de experimentos e modelos no MLflow.
- API de inferencia com validacao de entrada.
- Logs estruturados e metricas de saude.
- Monitoramento de drift, latencia, erros e custo de negocio.

## Riscos

- Drift de dados por mudancas comerciais ou perfil dos clientes.
- Desbalanceamento da classe de churn.
- Possiveis disparidades por segmentos, como senioridade, contrato ou forma de pagamento.
- Uso indevido como decisao automatizada sem supervisao humana.
