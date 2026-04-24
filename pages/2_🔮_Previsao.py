import streamlit as st
import numpy as np
import joblib
from tensorflow import keras

st.title("🔮 Previsão de Risco")

# carregar modelo
modelo = keras.models.load_model("model/modelo.h5")
scaler = joblib.load("model/scaler.pkl")

st.subheader("Insira os dados do aluno")

idade = st.slider("Idade", 15, 22, 17)
faltas = st.slider("Faltas", 0, 50, 5)
tempo_estudo = st.slider("Tempo de estudo", 1, 4, 2)
nota1 = st.slider("Nota período 1", 0, 20, 10)
nota2 = st.slider("Nota período 2", 0, 20, 10)

if st.button("Prever"):
    entrada = np.array([[0, 0, idade, 0, 0, 0, 2, 2, 0, 0, 0, 0, 2,
                         tempo_estudo, 0, 0, 0, 0, 0, 0, 1, 1, 0,
                         3, 3, 3, 1, 1, 3, faltas,
                         nota1, nota2]])

    entrada = scaler.transform(entrada)

    prob = modelo.predict(entrada)[0][0] * 100

    st.metric("Probabilidade de risco", f"{prob:.1f}%")

    if prob > 60:
        st.error("ALTO RISCO")
    elif prob > 40:
        st.warning("RISCO MODERADO")
    else:
        st.success("BAIXO RISCO")