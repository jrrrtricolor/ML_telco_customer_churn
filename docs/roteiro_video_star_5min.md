# Roteiro de Video (5 min) - Metodologia STAR

Tempo total alvo: 5 minutos
Contexto: projeto ML Telco Customer Churn
Formato: apresentacao oral com apoio de 1 slide por secao

## 0:00-0:20 | Abertura

Fala sugerida:
"Ola, nos somos Júlia e Cássio, equipe responsavel por este projeto. Neste video, vamos apresentar nosso projeto de previsao de churn em telecom usando a metodologia STAR: Situacao, Tarefa, Acao e Resultados. O objetivo foi construir uma solucao de machine learning ponta a ponta, desde dados ate inferencia por API."

Visual sugerido:
- Nome dos integrantes do grupo
- Titulo do projeto
- Agenda STAR em 4 blocos

---

## 0:20-1:20 | S - Situacao

Fala sugerida:
"A situacao de negocio e clara: a operadora perde clientes por cancelamento de contrato, o churn, e isso reduz receita recorrente. O desafio nao e apenas prever churn, mas priorizar quais clientes devem receber acao de retencao."

"Trabalhamos com o dataset Telco Customer Churn, com uma linha por cliente e variaveis de perfil, contrato, servicos e cobranca. A variavel alvo e Churn, tratada como classificacao binaria."

"Desde o inicio, consideramos que o problema tem impacto financeiro assimetrico: deixar passar um cliente que realmente cancelaria custa muito mais do que acionar um cliente que nao cancelaria."

Pontos para slide:
- Problema: perda de receita por churn
- Dados: cadastro + contrato + servicos + cobranca
- Alvo: Churn Yes/No
- Contexto operacional: priorizacao de campanhas de retencao

---

## 1:20-2:05 | T - Tarefa

Fala sugerida:
"Nossa tarefa como grupo foi entregar um pipeline completo, reproduzivel e avaliavel, com foco tecnico e de negocio."

"Os objetivos tecnicos foram: preparar os dados, treinar modelos baseline e MLP, comparar desempenho com metricas adequadas para classe desbalanceada, registrar experimentos no MLflow e disponibilizar inferencia via FastAPI com validacao de entrada, observabilidade e testes."

"Tambem definimos um objetivo de leitura de resultado: nao escolher modelo por accuracy isolada, e sim por equilibrio entre PR AUC, recall e custo de negocio."

Pontos para slide:
- Pipeline end-to-end
- Baselines + MLP
- MLflow para rastreabilidade
- API para inferencia
- Testes e monitoramento basico

---

## 2:05-4:10 | A - Acao

Fala sugerida:
"Na Acao, tomamos decisoes tecnicas em tres frentes: dados, modelagem e deploy/monitoramento."

"Em dados, removemos customerID, tratamos tipos numericos e limpamos inconsistencias como MonthlyCharges menor ou igual a zero. Para TotalCharges ausente, usamos imputacao baseada em tenure vezes MonthlyCharges. Depois, mapeamos Churn para 0 e 1 e fizemos split estratificado entre treino e teste para preservar a proporcao da classe alvo."

"Em modelagem, usamos um ModelFactory para padronizar os experimentos com Dummy, Decision Tree, Random Forest, KNN, Logistic Regression e MLP. Todos os modelos passaram por pipeline sklearn com preprocessamento consistente."

"No treinamento e avaliacao, aplicamos validacao cruzada estratificada de 5 folds no treino e calculamos accuracy, precision, recall, F1, ROC AUC e PR AUC. Alem disso, usamos uma metrica de negocio explicita: custo_negocio = FP vezes 100 mais FN vezes 840, para refletir impacto financeiro real."

"Para rastreabilidade, registramos parametros, metricas e artefatos no MLflow local, inclusive com registro de modelos. Isso permite comparar execucoes e versionar modelos."

"Em deploy, escolhemos FastAPI com Docker para uma entrega simples e reproduzivel. A API expoe health check e predicao, com validacao via Pydantic, logging estruturado e metricas Prometheus para latencia, volume de requisicoes e confianca media da predicao."

"Por fim, cobrimos qualidade com testes unitarios, smoke test da API, teste de schema com Pandera e teste e2e em Docker."

Pontos para slide:
- Limpeza e split estratificado
- Comparacao de baselines + MLP
- CV estratificada
- Metrica tecnica + metrica de negocio
- MLflow, FastAPI, Prometheus, testes

---

## 4:10-4:50 | R - Resultados

Fala sugerida:
"Nos resultados, tivemos uma base robusta e reproduzivel para previsao de churn. No snapshot documentado, Logistic Regression liderou em ROC AUC, PR AUC e tambem no menor custo de negocio. A MLP ficou competitiva frente aos baselines, cumprindo o objetivo de validar uma abordagem neural no contexto do desafio."

"Tambem entregamos o fluxo completo funcionando: treino, avaliacao, registro em MLflow, API para inferencia e monitoramento inicial."

Pontos para slide:
- Melhor desempenho geral no snapshot: Logistic Regression
- MLP competitiva
- Entrega end-to-end operacional

---

## 4:50-5:00 | Fechamento e Licoes Aprendidas

Fala sugerida:
"As principais licoes foram: primeiro, baseline forte precisa ser referencia antes de aumentar complexidade; segundo, metrica tecnica sozinha nao basta, custo de negocio muda a decisao; terceiro, rastreabilidade e observabilidade sao essenciais para evoluir o projeto com seguranca."

"Como proximo passo, vamos aprofundar fairness por segmento e monitoramento de drift para sustentar ganhos em uso continuo."

Pontos para slide:
- Baseline antes de complexidade
- Decisao orientada a negocio
- Proximos passos: fairness e drift
