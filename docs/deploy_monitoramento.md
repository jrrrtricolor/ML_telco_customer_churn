# Deploy e Monitoramento

## Arquitetura escolhida

O MVP usa inferência online simples com FastAPI.

Fluxo:

1. O pipeline treina os modelos e registra os artefatos no MLflow local.
2. A API carrega o modelo `churn_mlp` pelo registry do MLflow.
3. Caso o registry não esteja disponível, a API tenta usar o artefato local mais recente em `mlruns`.
4. O endpoint `/predict` recebe os dados do cliente e retorna a classe prevista.

## Justificativa

Para o Tech Challenge, a API em tempo real facilita a demonstração no vídeo e valida o ciclo completo de produtização. Em um cenário real, o primeiro uso recomendado ainda seria batch diário ou semanal para gerar listas de retenção para CRM.

## Estratégia de deploy

Opção acadêmica/local:

- Treinar com `make train`.
- Subir API com `make api`.
- Testar `/health` e `/predict`.

Opção em nuvem:

- Gerar imagem com o `Dockerfile`.
- Publicar a imagem em um registry.
- Subir em Azure Container Apps, AWS ECS/Fargate ou GCP Cloud Run.
- Configurar `MODEL_URI` quando o artefato do MLflow estiver fora da imagem.

## Monitoramento

Métricas técnicas:

- Latência média e p95 do `/predict`.
- Taxa de erro HTTP 4xx/5xx.
- Distribuição dos scores e classes previstas.
- Drift de variáveis importantes (`Contract`, `InternetService`, `MonthlyCharges`, `tenure`).

Métricas de negócio:

- Taxa de churn dos clientes abordados.
- Custo de retenção.
- Receita potencial preservada.
- Comparação entre clientes abordados e grupo de controle.

## Alertas sugeridos

- Latência p95 acima de 500 ms por 15 minutos.
- Erro 5xx acima de 2% em uma janela de 10 minutos.
- Mudança relevante na distribuição de uma feature crítica.
- Queda de recall em validação posterior abaixo de 0.50.

## Playbook inicial

1. Confirmar se a API está saudável em `/health`.
2. Verificar logs de latência e erro.
3. Validar se o modelo correto foi carregado.
4. Comparar dados recentes com o schema esperado.
5. Se houver drift ou queda de performance, pausar uso operacional e retreinar.
