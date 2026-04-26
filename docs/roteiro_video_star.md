# Roteiro do Video STAR

Tempo alvo: ate 5 minutos.

## Situation

Uma operadora de telecomunicacoes esta perdendo clientes e precisa priorizar acoes de retencao. O problema foi tratado como classificacao binaria de churn usando o dataset Telco Customer Churn, com dados cadastrais, contratuais, servicos contratados e informacoes de cobranca.

## Task

O objetivo do grupo foi construir um pipeline profissional end-to-end: preparar dados, treinar baselines, construir uma MLP em PyTorch, comparar resultados com metricas tecnicas e de negocio, registrar experimentos no MLflow e servir o modelo por API.

## Action

O projeto foi organizado em modulos dentro de `src/`, com pipeline de limpeza, split estratificado, transformadores sklearn, baselines e MLP. A rede neural usa PyTorch, mini-batches, validacao interna e early stopping. As metricas incluem accuracy, precision, recall, F1, ROC AUC, PR AUC e custo de negocio baseado em falso positivo e falso negativo.

Os experimentos sao registrados no MLflow. A inferencia local e exposta por FastAPI com validacao Pydantic, endpoints `/health` e `/predict`, logging estruturado e metricas Prometheus. Tambem foram adicionados testes unitarios, teste de schema com Pandera, smoke test da API e Makefile para padronizar comandos.

## Result

O projeto entrega uma base reprodutivel para prever clientes com risco de churn. A MLP apresentou bom resultado de custo de negocio no snapshot documentado, enquanto modelos como Random Forest ajudaram na comparacao por AUC. As principais licoes foram a importancia de comparar modelos simples contra redes neurais, avaliar custo de erro e manter rastreabilidade de dados, metricas e artefatos.

Para evolucao, o proximo passo e fortalecer a analise por segmentos e acompanhar drift, latencia, erros e impacto de negocio. Deploy em nuvem fica como bonus opcional fora do escopo atual.
