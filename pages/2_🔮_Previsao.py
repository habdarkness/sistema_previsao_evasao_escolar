import streamlit as st
import numpy as np
import pandas as pd
import joblib
from tensorflow import keras
from utils.preprocess import load_data, classificate, encode, get_features

st.set_page_config(page_title="Previsão de Risco", layout="centered")
st.title("🔮 Previsão de Risco Escolar")

# =========================
# CARREGAR MODELO
# =========================
modelo = keras.models.load_model("model/modelo.h5")
scaler = joblib.load("model/scaler.pkl")
encoders = joblib.load("model/encoders.pkl")

# =========================
# CARREGAR DADOS BASE
# =========================
df = load_data()
df, encoders = encode(df)

features = get_features()

# =========================
# INPUTS
# =========================
st.subheader("📊 Insira os dados do aluno")

idade = st.slider("Idade", 15, 22, 17)
genero = st.selectbox("Gênero", ["M", "F"])
faltas = st.slider("Faltas", 0, 50, 5)
tempo_estudo = st.slider("Tempo de estudo", 1, 4, 2)
nota1 = st.slider("Nota período 1", 0, 20, 10)
nota2 = st.slider("Nota período 2", 0, 20, 10)

# =========================
# PREVISÃO
# =========================
if st.button("Prever"):

    with st.spinner("Calculando previsão... 🔄"):
        base = df.iloc[0].copy()
        
        base["idade"] = idade
        base["faltas"] = faltas
        base["tempo_estudo"] = tempo_estudo
        base["nota_periodo_1"] = nota1
        base["nota_periodo_2"] = nota2
        base["sexo"] = encoders["sexo"].transform([genero])[0]

        entrada = base[features].values.reshape(1, -1)
        entrada = scaler.transform(entrada)

        prob = modelo.predict(entrada)[0][0] * 100

    # =========================
    # RESULTADO
    # =========================
    st.subheader("📊 Classificação do aluno")
    st.metric("Probabilidade de risco", f"{prob:.1f}%")
    nivel = classificate(prob)
    # cores
    if "CRITICO" in nivel:
        st.error(nivel)
    elif "RUIM" in nivel:
        st.warning(nivel)
    elif "REGULAR" in nivel:
        st.info(nivel)
    else:
        st.success(nivel)