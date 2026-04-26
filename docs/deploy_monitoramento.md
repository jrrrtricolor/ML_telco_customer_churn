# Deploy e Monitoramento

## Decisao de arquitetura

O projeto usa inferencia online como modo de demonstracao principal, servida por FastAPI local/Docker. O deploy em nuvem e um bonus opcional do Tech Challenge e nao faz parte do escopo atual da entrega.

A escolha por FastAPI + Docker mantem a entrega simples, reproduzivel e aderente aos requisitos obrigatorios: API funcional, validacao de entrada, logging estruturado, testes e monitoramento basico. O pipeline segue registrando modelos no MLflow local para rastreabilidade de experimentos e artefatos.

## Alternativas avaliadas

- Batch: adequado para campanhas de retencao periodicas e menor custo operacional.
- Real-time API: adequado para demonstracao, integracao com CRM e consultas sob demanda.
- Nuvem gerenciada: fora do escopo atual, mantida como evolucao futura caso o bonus seja perseguido.

## Plano de deploy local/Docker

1. Executar o treino em ambiente com dependencias instaladas.
2. Registrar o melhor modelo no MLflow com `registered_model_name`.
3. Subir a API local com `make api` ou por Docker.
4. Validar `/health`, `/predict` e `/metrics`.
5. Executar smoke test e, quando necessario, teste e2e Docker.
6. Documentar a versao do modelo registrada no MLflow.

## Monitoramento

Metricas tecnicas:

- Latencia de predicao.
- Taxa de erro HTTP.
- Volume de requisicoes.
- Confianca media das predicoes.
- Distribuicao dos scores de churn.

Metricas de modelo:

- Precision, recall, F1, ROC AUC e PR AUC em dados rotulados recentes.
- Custo de negocio estimado por falso positivo e falso negativo.
- Drift de entrada por variavel critica.
- Drift de saida na distribuicao de scores.

Metricas de negocio:

- Taxa de churn mensal.
- Clientes acionados por campanha.
- Conversao das acoes de retencao.
- Receita preservada estimada.

## Alertas

- Latencia media acima de 1 segundo por 5 minutos.
- Erros de API acima do limite operacional definido.
- Queda de confianca media abaixo de 0.5.
- Drift relevante em variaveis como `Contract`, `tenure`, `MonthlyCharges` e `InternetService`.
- Aumento de falsos negativos em amostras rotuladas.

## Playbook de resposta

1. Verificar saude do endpoint e logs da API.
2. Confirmar se houve mudanca recente de versao do modelo.
3. Comparar distribuicao dos dados atuais com o dataset de treino.
4. Se houver drift, pausar automacoes de alto impacto e acionar revisao humana.
5. Reprocessar dados recentes, reavaliar modelos e registrar nova execucao no MLflow.
6. Publicar nova versao somente apos comparar metricas tecnicas e custo de negocio.
