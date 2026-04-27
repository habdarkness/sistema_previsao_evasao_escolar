import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="Sistema de Previsão Escolar",
    layout="wide"
)

# =========================
# CACHE (PERFORMANCE)
# =========================
@st.cache_data
def load_info():
    metricas = pd.read_csv("data/metricas_modelo.csv", index_col=0)
    conf = pd.read_csv("data/matriz_confusao.csv", index_col=0)
    resumo = pd.read_csv("data/resumo_modelo.csv")
    previsoes = pd.read_csv("data/previsoes_alunos.csv")
    return metricas, conf, resumo, previsoes

df_metricas, df_conf, df_resumo, df = load_info()

# =========================
# TÍTULO
# =========================
st.title("📘 Sistema Inteligente de Previsão Escolar")

st.markdown("""
Este sistema utiliza **Redes Neurais Artificiais** para prever o risco de desempenho escolar.

### 🎯 Objetivo
Identificar alunos em risco para permitir intervenções antecipadas.

### 🧠 Tecnologias utilizadas
- TensorFlow / Keras
- Scikit-learn
- Streamlit

Use o menu lateral para navegar entre as páginas.
""")

st.divider()

# =========================
# INSIGHTS
# =========================
st.subheader("🧠 Insights do Modelo")

st.markdown("""
- O modelo prioriza **não deixar alunos em risco passarem despercebidos**
- Há tendência de **marcar mais alunos como risco (segurança > precisão)**

### ⚠️ Limitações
- Pode gerar falsos positivos
- Não substitui avaliação pedagógica real
""")

st.divider()

# =========================
# MÉTRICAS
# =========================
st.header("📊 Desempenho do Modelo")
col1, col2, col3, col4 = st.columns(4)

acuracia = df_metricas.loc["accuracy", "precision"] * 100
precisao_alto = df_metricas.loc["Alto Risco", "precision"] * 100
recall_alto = df_metricas.loc["Alto Risco", "recall"] * 100
f1_alto = df_metricas.loc["Alto Risco", "f1-score"] * 100

col1.metric("Acurácia", f"{acuracia:.1f}%")
col2.metric("Precisão (Alto risco)", f"{precisao_alto:.1f}%")
col3.metric("Recall (Alto risco)", f"{recall_alto:.1f}%")
col4.metric("F1-Score", f"{f1_alto:.1f}%")

st.markdown("""
**Objetivo:** Exibir os principais indicadores de desempenho do modelo e detalhar a qualidade das previsões para cada classe de risco.
""")

st.divider()
# =========================
# MATRIZ DE CONFUSÃO
# =========================
st.subheader("📉 Matriz de Confusão")

fig = px.imshow(
    df_conf,
    text_auto=True,
    aspect="auto",
    labels=dict(x="Previsto", y="Real", color="Quantidade")
)

st.plotly_chart(fig, use_container_width=True)

st.divider()

# =========================
# RESUMO
# =========================
st.subheader("📊 Distribuição geral")

col1, col2 = st.columns(2)

col1.metric("Total de alunos", int(df_resumo["total_alunos"][0]))
col2.metric("Alunos em risco", int(df_resumo["alunos_risco"][0]))

st.divider()

# =========================
# TABELA FINAL
# =========================
st.subheader("📋 Banco de dados")

st.dataframe(df.head(10))