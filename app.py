import streamlit as st

st.set_page_config(
    page_title="Sistema de Previsão Escolar",
    layout="wide"
)

st.title("📘 Sistema Inteligente de Previsão Escolar")

st.markdown("""
Este sistema utiliza **Redes Neurais Artificiais** para prever o risco de desempenho escolar.

### 🎯 Objetivo
Identificar alunos em risco para permitir intervenções antecipadas.

### 🧠 Tecnologias utilizadas
- TensorFlow / Keras
- Scikit-learn
- Streamlit

### 📊 Funcionalidades
- Análise de dados com gráficos
- Filtros interativos
- Previsão de risco em tempo real

Use o menu lateral para navegar entre as páginas.
""")