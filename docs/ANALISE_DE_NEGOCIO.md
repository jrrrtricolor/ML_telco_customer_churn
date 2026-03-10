# 1. Resumo tecnico executivo

## Problema
O projeto tem como objetivo prever a probabilidade de churn de clientes de telecom para priorizar acoes preventivas de retencao antes do cancelamento. O problema e de classificacao binaria supervisionada, com variavel alvo `Churn` (`Yes/No` ou `1/0`), e sera operacionalizado por meio de uma API HTTP de inferencia online, aceitando estritamente 1 registro por chamada, sem processamento batch.

O resultado esperado para o negocio e reduzir cancelamentos evitaveis por meio de acionamento orientado por score de risco, com rastreabilidade de decisoes e monitoramento continuo de desempenho tecnico e impacto financeiro.

# 2. Escopo e criterio de sucesso

## Pergunta analitica principal
Qual a probabilidade de um cliente ativo cancelar o servico no horizonte de curto prazo, com qualidade preditiva suficiente para suportar campanhas de retencao com retorno financeiro positivo?

## Definicao de sucesso de negocio (KPI de retencao, custo evitado, uplift esperado)
- KPI de retencao: reduzir a taxa de churn mensal em pelo menos 8% relativo no segmento alvo das acoes preventivas em ate 90 dias apos entrada em producao.
- Custo evitado: gerar economia acumulada minima de R$ 300.000,00 em 6 meses por reducao de cancelamentos evitaveis, descontado custo de acao de retencao.
- Uplift esperado: obter uplift minimo de 12 pontos percentuais na taxa de permanencia entre clientes de alto risco tratados vs. grupo de controle comparavel.

## Definicao de sucesso tecnico (metricas alvo)
- Acuracia minima: 0,80 em validacao temporal.
- Precisao minima (classe churn): 0,70 para controlar custo de abordagens desnecessarias.
- Recall minimo (classe churn): 0,78 para reduzir perda de clientes de alto risco nao acionados.
- F1-score minimo (classe churn): 0,74 para balancear precisao e cobertura.
- Latencia de API: p95 <= 200 ms por requisicao.
- Disponibilidade de API: >= 99,5% mensal.
- Taxa de erro de API (5xx): <= 0,5% mensal.

## Prazo e orcamento do projeto
- Prazo final comprometido: `05/05/2026`.
- Janela de execucao considerada: marco a maio de 2026, com entregas incrementais e entrada em producao ate a data-limite.
- Orcamento total estimado: `R$ 286.000,00`.
- Composicao do orcamento - Desenvolvimento de ML e pipeline de inferencia online (4 engenheiros, periodo do projeto): R$ 230.000,00.
- Composicao do orcamento - Infraestrutura (hosting da API, observabilidade, armazenamento e CI/CD): R$ 24.000,00.
- Composicao do orcamento - Governanca de dados, seguranca e conformidade LGPD (apoio juridico/compliance): R$ 12.000,00.
- Composicao do orcamento - Reserva de risco operacional e tecnico: R$ 20.000,00.

# 3. Dados disponiveis e LGPD

## Dados disponiveis para o escopo
- Dados do cliente: `customerID`, `gender`, `SeniorCitizen`, `Partner`, `Dependents`, `tenure`.
- Servicos contratados: `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`.
- Informacoes contratuais e financeiras: `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges`.
- Variavel alvo: `Churn`.

## Restricoes e controles LGPD aplicaveis
- Dados pessoais identificaveis: `customerID` (identificador de cliente), dados demograficos e contratuais associados ao titular.
- Dados de perfil financeiro/comportamental: `PaymentMethod`, `MonthlyCharges`, `TotalCharges` e historico de servicos podem caracterizar perfil de consumo e exigem controles reforcados.
- Nao ha, no escopo informado, dado pessoal sensivel nos termos estritos da LGPD; ainda assim, `SeniorCitizen` exige avaliacao de risco de vies e discriminacao indireta.
- Principios obrigatorios no tratamento: finalidade especifica (retencao), adequacao, necessidade (minimizacao), transparencia, seguranca e prevencao.
- Medidas minimas de conformidade - Pseudonimizacao de `customerID` em ambientes analiticos.
- Medidas minimas de conformidade - Criptografia em transito (TLS) e em repouso para dados e artefatos.
- Medidas minimas de conformidade - Controle de acesso por privilegio minimo e trilha de auditoria.
- Medidas minimas de conformidade - Politica de retencao e descarte de dados alinhada ao prazo de negocio.
- Medidas minimas de conformidade - Registro de base legal e avaliacao de impacto quando aplicavel.
- Restricao operacional de privacidade: inferencia online com 1 registro por chamada reduz exposicao de dados em lote e deve ser mantida como requisito nao funcional.

# 4. Lista de stakeholders

- Diretoria de Negocio/Receita: patrocinio, metas de retencao e aprovacao de ROI.
- Planejamento Comercial e CRM: definicao de segmentos, campanhas e regras de acionamento.
- Atendimento/Call Center: execucao das acoes de retencao e feedback operacional.
- Engenharia de Dados: garantia de disponibilidade, qualidade e linhagem dos dados.
- Engenharia de Software/API: disponibilizacao e sustentacao da API HTTP online.
- MLOps/SRE: observabilidade, deploy, monitoramento de drift e SLOs de producao.
- Seguranca da Informacao e Compliance/Juridico: governanca LGPD e requisitos de seguranca.
- Controladoria/Financeiro: validacao de custo evitado e acompanhamento de beneficios economicos.

# 5. Lista de engenheiros

- Cassio
- Julia
- Lara
- Rayane

# 6. Riscos

- Risco de baixa acuracia em segmentos especificos; impacto: queda de efetividade das campanhas e aumento de custo por contato improdutivo; mitigacao: validacao estratificada por perfil de contrato/servico e monitoramento continuo de metricas por segmento.
- Risco de degradacao de desempenho do modelo ao longo do tempo (data/concept drift); impacto: perda progressiva de recall e aumento de churn nao prevenido; mitigacao: monitoramento em producao, gatilhos de reavaliacao e rotina de retreinamento controlado.
- Risco de indisponibilidade ou latencia acima do acordado na API online; impacto: indisponibilidade de score em tempo habil para acao de retencao; mitigacao: arquitetura resiliente, observabilidade com alerta proativo e SLOs com plano de contingencia.
- Risco de nao conformidade LGPD; impacto: passivo regulatorio, reputacional e possivel restricao de uso de dados; mitigacao: privacy by design, trilha de auditoria, controle de acesso e revisao periodica de conformidade.
- Risco de desalinhamento entre score tecnico e estrategia comercial; impacto: baixo uplift mesmo com modelo tecnicamente adequado; mitigacao: governanca interdepartamental com ritos de calibracao entre negocio, CRM e times tecnicos.
