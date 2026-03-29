# ML Canvas - Previsão de Churn (Phase Zero)

Referência: https://ml-ops.org/content/phase-zero

## 1. Proposta de Valor
- Problema: perda de clientes reduz receita recorrente e aumenta custo de aquisição.
- Objetivo: reduzir churn mensal com ações de retenção orientadas por risco.
- Usuário principal: Retenção, CRM e Atendimento.
- Valor entregue: priorização de clientes com maior probabilidade de cancelamento.

## 2. Fontes de Dados
- Fonte principal: data/raw/Telco_Customer_Churn.csv.
- Unidade de análise: 1 linha por cliente.
- Riscos de qualidade: nulos em TotalCharges e possível desbalanceamento da classe.
- Restrição atual: sem uso de dados externos nesta fase.

## 3. Tarefa de Predição
- Tipo: classificação binária supervisionada.
- Entrada: dados cadastrais, contratuais, de serviços e financeiros.
- Saída: probabilidade de churn no próximo ciclo.
- Rótulo: Yes = 1, No = 0.

## 4. Atributos (Engenharia)
- Principais grupos: perfil, contrato, serviços e cobrança.
- Tratamento previsto: limpeza de nulos, padronização de categorias e encoding.
- Exclusão: customerID não entra no modelo.

## 5. Avaliação Offline
- Métricas alvo:
  - Recall >= 0.70
  - ROC-AUC >= 0.75
  - Accuracy > 0.70
- Validação: divisão estratificada em treino, validação e teste.
- Custo de erro:
  - FN: maior risco de perda de receita.
  - FP: maior custo de campanha.

## 6. Decisões
- Uso da predição: priorizar clientes para campanha de retenção.
- Regra operacional: clientes acima do limiar entram na fila de ação.
- Responsável pelo limiar: time de CRM.

## 7. Fazer Predições
- Modo inicial: batch diário (ou semanal no piloto).
- Entrega: arquivo de score para consumo em CRM/BI.
- Requisito: disponibilizar antes da janela de campanha.

## 8. Coleta de Dados
- Novos dados: resposta a campanhas e eventos de atendimento.
- Rotulagem: churn confirmado no fechamento do ciclo.
- Controle: monitorar completude e drift de dados.

## 9. Construir Modelos
- Retreino inicial: mensal.
- Governança: versionamento de artefatos, parâmetros e experimentos.
- Custo: execução local no piloto; estimativa de nuvem em fase futura.

## 10. Avaliação em Produção e Monitoramento
- Métricas de negócio: taxa de churn, receita preservada e uplift.
- Métricas de modelo: estabilidade de score, precision, recall e drift.
- Alerta: queda recorrente de desempenho por 2 ciclos.

## Riscos e Gate de Viabilidade
- Qualidade de dados insuficiente.
- ROI baixo quando custo de campanha supera retorno.
- Projeto avança somente com metas técnicas e de negócio atendidas.

## Escopo do MVP
- Inclui: preparo de dados, treino, avaliação offline e ranking de risco.
- Não inclui: inferência online em tempo real e automação completa em nuvem.
