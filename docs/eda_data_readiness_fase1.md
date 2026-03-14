# EDA e Data Readiness - Fase 1

## 1. Base analisada

- arquivo: `data/raw/Telco_Customer_Churn.csv`
- linhas: 7043
- colunas: 21

## 2. Volume e estrutura

- granularidade: 1 linha por cliente
- coluna identificadora: `customerID` (alta cardinalidade, removida da modelagem)
- alvo: `Churn` (binario)

## 3. Qualidade dos dados

- duplicados completos: 0
- nulos explicitos por coluna: 0
- problema identificado: 11 registros com `TotalCharges` em branco (string vazia)
- tratamento adotado no pipeline:
  - conversao para numerico com `errors="coerce"`
  - imputacao orientada por regra: `TotalCharges = MonthlyCharges * max(tenure, 1)` para casos invalidos

## 4. Distribuicao das variaveis

### 4.1 Distribuicao do target

- `Churn = No`: 5174 (73.46%)
- `Churn = Yes`: 1869 (26.54%)

Leitura: dataset com desbalanceamento moderado, mas ainda viavel para baseline supervisionado sem tecnicas complexas iniciais.

### 4.2 Sumario de variaveis numericas

- `tenure`: min 0, mediana 29, media 32.37, max 72
- `MonthlyCharges`: min 18.25, mediana 70.35, media 64.76, max 118.75
- `TotalCharges`: min 18.8, mediana 1397.48, media 2283.30, max 8684.8

### 4.3 Cardinalidade de categoricas (destaques)

- `PaymentMethod`: 4 categorias
- `Contract`: 3 categorias
- `InternetService`: 3 categorias
- variaveis binarias predominantes: genero, parceiro, dependentes, paperless billing

## 5. Data readiness

Status por criterio:

- completude: parcial (ajuste necessario em `TotalCharges`)
- consistencia de tipos: adequada apos casting numerico
- disponibilidade de target: adequada
- risco de leakage: controlado via preprocessamento em pipeline apos split
- capacidade de modelagem baseline: adequada

Conclusao: dataset pronto para Fase 1 com tratamento de qualidade aplicado no codigo de preprocessamento.

## 6. Riscos residuais

- desbalanceamento pode reduzir recall de churn em modelos conservadores
- premissas financeiras da metrica de negocio ainda sao iniciais e devem ser calibradas com negocio
