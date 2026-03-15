# 1. Resumo tecnico executivo

## Problema
O projeto trata da previsao de churn de clientes de telecom por meio de classificacao binaria supervisionada, com saida de probabilidade de cancelamento por cliente.

O produto de dados sera uma API HTTP de inferencia online, com processamento de apenas 1 registro por chamada e sem execucoes em batch. O score de risco deve ser consumido por fluxos de retencao para acionar acoes preventivas antes do cancelamento.

Prazo de entrega: 05/05/2026.

Orcamento estimado do projeto (ate o go-live):
- Faixa total: R$ 198.000 a R$ 252.000.
- Pessoas (time tecnico e gestao): R$ 152.000 a R$ 192.000.
- Infraestrutura de desenvolvimento, homologacao e producao: R$ 28.000 a R$ 36.000.
- Observabilidade, seguranca e contingencia operacional: R$ 18.000 a R$ 24.000.

# 2. Escopo e criterio de sucesso

## Pergunta analitica principal
Qual a probabilidade de um cliente ativo cancelar o servico no curto prazo, considerando perfil do cliente, servicos contratados e caracteristicas contratuais, para priorizar intervencoes de retencao com melhor retorno operacional?

## Definicao de sucesso de negocio
KPIs de negocio para acompanhamento apos entrada em producao:
- KPI de retencao: reduzir churn mensal em pelo menos 1,5 p.p. no publico acionado em ate 90 dias apos go-live.
- KPI de custo evitado: gerar custo evitado minimo de R$ 350.000 por trimestre pela reducao de cancelamentos no publico tratado.
- KPI de uplift esperado: atingir uplift minimo de 12% na taxa de retencao das campanhas orientadas por score versus grupo controle.
- KPI de eficiencia operacional: elevar a taxa de contato util (clientes de risco alto efetivamente abordados) para pelo menos 85% da capacidade mensal da operacao de retencao.

## Definicao de sucesso tecnico (metricas alvo)
Metas minimas para aprovacao tecnica do modelo e da API:
- Acuracia global: >= 0,80.
- Precisao para classe churn: >= 0,72.
- Recall para classe churn: >= 0,78.
- F1-score para classe churn: >= 0,75.
- Latencia de API (p95): <= 150 ms por requisicao em ambiente de producao.
- Disponibilidade de API (mensal): >= 99,5%.
- Taxa de erro de API (HTTP 5xx): <= 0,5% das requisicoes.

## Conexao entre sucesso tecnico e sucesso de negocio
Com recall de 78%, a solucao identifica 78 em cada 100 clientes que efetivamente cancelariam, ampliando cobertura de risco para campanhas de retencao.

Com precisao de 72%, cerca de 72 em cada 100 clientes sinalizados como alto risco tendem a ser casos relevantes, reduzindo desperdicio de incentivos e carga improdutiva na operacao.

Com latencia p95 de ate 150 ms e disponibilidade de 99,5%, a API suporta uso online em canais de atendimento e CRM sem bloquear jornada de decisao comercial, mantendo a atuacao preventiva em tempo habil.

Escopo operacional obrigatorio:
- Inferencia exclusivamente online via API HTTP.
- Apenas 1 registro por chamada.
- Proibido processamento em batch para inferencia.

# 3. Dados disponiveis e LGPD

## Dados disponiveis
Campos informados para uso no projeto:
- Dados do cliente: `customerID`, `gender`, `SeniorCitizen`, `Partner`, `Dependents`, `tenure`.
- Servicos contratados: `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`.
- Informacoes contratuais e cobranca: `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges`.
- Variavel alvo: `Churn`.

## Restricoes e requisitos LGPD
A base contem dados pessoais e dados comportamentais vinculados a pessoa natural identificavel por `customerID` (identificador direto no contexto da operadora). O tratamento deve observar finalidade, necessidade e seguranca.

Classificacao pratica para governanca:
- Dado pessoal de identificacao: `customerID`.
- Dados pessoais cadastrais e perfil: `gender`, `SeniorCitizen`, `Partner`, `Dependents`.
- Dados de relacao contratual e consumo de servicos: `tenure`, servicos contratados, `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges`.
- Dado comportamental de resultado: `Churn`.

Controles minimos exigidos:
- Minimizar uso de atributos em inferencia e monitoramento ao estritamente necessario para previsao.
- Pseudonimizar identificadores em ambientes analiticos e de teste.
- Restringir acesso por perfil (RBAC) para dados brutos e logs de inferencia.
- Criptografar dados em repouso e em transito.
- Registrar trilha de auditoria de inferencias com `request_id`, versao de modelo, timestamp e score; evitar persistencia de payload completo sem necessidade operacional.
- Definir politica de retencao e descarte para dados de treinamento, logs e artefatos.
- Validar base legal do tratamento com Juridico/Privacidade (execucao de contrato e/ou legitimo interesse, conforme processo de retencao adotado).

Ponto de atencao de conformidade:
- Mesmo sem dados pessoais sensiveis explicitos, o atributo `SeniorCitizen` requer cuidado adicional para evitar vies operacional e para garantir justificativa de uso alinhada ao objetivo de retencao.

# 4. Lista de stakeholders

- Diretoria de Negocios/Comercial.
- CRM e Retencao de Clientes.
- Marketing de Relacionamento e Campanhas.
- Operacao de Atendimento (Call Center e Canais Digitais).
- Engenharia de Dados e Plataforma.
- Ciencia de Dados e MLOps.
- TI/Arquitetura e SRE.
- Juridico, Privacidade e DPO.
- Financeiro e Controladoria.
- Governanca e Risco.

# 5. Lista de engenheiros

- Cassio
- Julia
- Lara
- Rayane

# 4. Riscos

- Risco de qualidade de dados de entrada (campos faltantes, padroes inconsistentes, mudancas de preenchimento) impactar score.
- Risco de drift de dados e de conceito reduzir performance tecnica e efetividade de retencao ao longo do tempo.
- Risco de indisponibilidade ou latencia acima da meta da API afetar operacao online de atendimento.
- Risco de baixo alinhamento operacional entre score e estrategia de campanha reduzir uplift real.
- Risco de nao conformidade LGPD por excesso de dados em logs ou acesso indevido.
- Risco de restricao de capacidade de contato da operacao impedir cobertura dos clientes de maior risco.
- Risco de prazo (05/05/2026) por dependencia entre dados, integracao de API e homologacao com areas de negocio.
- Risco de estouro de orcamento por aumento de custo de infraestrutura, retrabalho regulatorio ou ajustes de integracao.
