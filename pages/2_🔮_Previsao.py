import streamlit as st
import numpy as np
import pandas as pd
import joblib
from tensorflow import keras
from utils.preprocess import carregar_dados
from utils.preprocess import classificar_nivel

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Previsão de Risco", layout="centered")
st.title("🔮 Previsão de Risco Escolar")

# =========================
# CARREGAR MODELO
# =========================
modelo = keras.models.load_model("model/modelo.h5")
scaler = joblib.load("model/scaler.pkl")
encoders = joblib.load("model/encoders.pkl")

# =========================
# CARREGAR DADOS BASE (IGUAL TREINO)
# =========================
df = carregar_dados()

# aplicar os MESMOS encoders do treino
for col, le in encoders.items():
    df[col] = le.transform(df[col].astype(str))

# mesmas features do modelo
features = [
    "escola", "sexo", "idade", "tipo_residencia", "tamanho_familia",
    "situacao_pais", "educacao_mae", "educacao_pai", "trabalho_mae",
    "trabalho_pai", "motivo_escola", "responsavel", "tempo_viagem",
    "tempo_estudo", "reprovacoes_anteriores", "apoio_escola", "apoio_familia",
    "aulas_pagas", "atividades_extracurriculares", "frequentou_creche",
    "deseja_ensino_superior", "acesso_internet", "relacionamento_romantico",
    "qualidade_relacoes_familiares", "tempo_livre", "sair_com_amigos",
    "consumo_alcool_semana", "consumo_alcool_fds", "saude", "faltas",
    "nota_periodo_1", "nota_periodo_2",
]

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

        # pega um aluno base REAL (já tratado)
        base = df.iloc[0].copy()

        # substitui valores do usuário
        base["idade"] = idade
        base["faltas"] = faltas
        base["tempo_estudo"] = tempo_estudo
        base["nota_periodo_1"] = nota1
        base["nota_periodo_2"] = nota2

        # 🔥 usa encoder correto (ESSENCIAL)
        base["sexo"] = encoders["sexo"].transform([genero])[0]

        # garante ordem correta
        entrada = base[features].values.reshape(1, -1)

        # normaliza
        entrada = scaler.transform(entrada)

        # previsão
        prob = modelo.predict(entrada)[0][0] * 100

    # =========================
    # RESULTADO
    # =========================
    st.subheader("📊 Classificação do aluno")
    st.metric("Probabilidade de risco", f"{prob:.1f}%")
    nivel = classificar_nivel(prob)
    # cores
    if "CRITICO" in nivel:
        st.error(nivel)
    elif "RUIM" in nivel:
        st.warning(nivel)
    elif "REGULAR" in nivel:
        st.info(nivel)
    else:
        st.success(nivel)