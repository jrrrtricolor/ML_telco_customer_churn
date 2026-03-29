# 📊 Definição de Métricas — Churn

## 🎯 Objetivo
Prever quais clientes têm maior chance de cancelar (churn), permitindo ações de retenção.

---

## 📈 Métricas Técnicas

Para avaliar o modelo, foram utilizadas:

- **AUC-ROC**  
  Mede a capacidade do modelo de separar clientes que cancelam dos que não cancelam.

- **PR-AUC (Principal)**  
  Foca na identificação correta dos clientes que realmente vão cancelar.  
  É mais adequada para bases desbalanceadas (como churn).

- **F1-score**  
  Equilibra:
  - Precisão (acertar quem vai cancelar)
  - Recall (não deixar passar quem vai cancelar)

---

## 💼 Métrica de Negócio

O objetivo não é apenas acertar previsões, mas **reduzir perdas financeiras**.

### 🔻 Tipos de erro:

- **Falso Negativo (FN)**  
  Cliente ia cancelar e não foi identificado → **perda de receita**

- **Falso Positivo (FP)**  
  Cliente não ia cancelar, mas recebeu oferta → **custo de retenção**

---

## 💰 Definição de custo

- Valor médio por cliente: **€840**
- Custo de retenção: **€100**

### Fórmula: 
Dicionário:
* FP = Falso Positivo
* FN = Falso Negativo
* 100 = custo de retenção por cliente
* 840 = perda média por cliente que cancela

Custo = (FP × 100) + (FN × 840)

---

## 🧠 Estratégia

- Priorizar a redução de **Falsos Negativos**
- Usar **PR-AUC como métrica principal**
- Avaliar modelos também pelo **impacto financeiro**

---

## ✅ Resumo

- Métrica principal: **PR-AUC**
- Métricas de apoio: **AUC-ROC e F1-score**
- Foco do modelo: **reduzir churn e minimizar perdas financeiras**