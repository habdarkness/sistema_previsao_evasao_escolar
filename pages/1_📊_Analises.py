import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Análises", layout="wide")

st.title("📊 Dashboard de Análise de Risco Escolar")

# =========================
# ORDEM DOS NÍVEIS
# =========================
ordem_niveis = [
    "CRITICO - INTERVENÇÃO PEDAGOGICA PROFUNDA",
    "RUIM - PLANO DE RECUPERAÇÃO URGENTE",
    "REGULAR - APOIO EXTRA OU TUTORIA",
    "BOM - MONITORIZAÇÃO OCASIONAL",
    "EXCELENTE - MANTER AUTONOMIA",
]

# =========================
# CORES
# =========================
cores_risco = {
    "EXCELENTE - MANTER AUTONOMIA": "#2ECC71",
    "BOM - MONITORIZAÇÃO OCASIONAL": "#90EE90",
    "REGULAR - APOIO EXTRA OU TUTORIA": "#9ACD32",
    "RUIM - PLANO DE RECUPERAÇÃO URGENTE": "#F39C12",
    "CRITICO - INTERVENÇÃO PEDAGOGICA PROFUNDA": "#E74C3C",
}

# =========================
# CARREGAR DADOS
# =========================
df = pd.read_csv("data/previsoes_alunos.csv")

df["nivel_risco"] = pd.Categorical(
    df["nivel_risco"],
    categories=ordem_niveis,
    ordered=True
)

# =========================
# FILTROS
# =========================
st.sidebar.header("Filtros")

nivel = st.sidebar.multiselect(
    "Nível de risco",
    ordem_niveis,
    default=ordem_niveis
)

gender = st.sidebar.multiselect(
    "Gênero",
    ["M", "F"],
    default=["M", "F"]
)

idade_min = st.sidebar.slider("Idade mínima", 15, 22, 15)
idade_max = st.sidebar.slider("Idade máxima", 15, 22, 22)

nota_min = st.sidebar.slider("Nota mínima", 0, 20, 0)
nota_max = st.sidebar.slider("Nota máxima", 0, 20, 20)

df_filtrado = df[
    (df["nivel_risco"].isin(nivel)) &
    (df["nota_final"] >= nota_min) &
    (df["nota_final"] <= nota_max) &
    (df["idade"] >= idade_min) &
    (df["idade"] <= idade_max) &
    (df["sexo"].isin(gender))
]

# =========================
# KPIs
# =========================
col1, col2, col3 = st.columns(3)

col1.metric("Total de alunos", len(df_filtrado))
col2.metric("Alunos em alto risco", int(df_filtrado["risco_previsto"].sum()))
col3.metric("Alunos em baixo risco", int((df_filtrado["risco_previsto"] == 0).sum()))

st.divider()

# =========================
# 1 - RISCO POR NÍVEL
# =========================
st.subheader("📊 Distribuição por nível de risco")

st.markdown("Quantidade de alunos em cada nível de risco.")

contagem = (
    df_filtrado["nivel_risco"]
    .value_counts()
    .reindex(ordem_niveis)
    .reset_index()
)

contagem.columns = ["nivel_risco", "quantidade"]

fig1 = px.bar(
    contagem,
    x="nivel_risco",
    y="quantidade",
    color="nivel_risco",
    labels={
        "nivel_risco": "Nível de risco",
        "quantidade": "Número de alunos"
    },
    category_orders={"nivel_risco": ordem_niveis},
    color_discrete_map=cores_risco
)

st.plotly_chart(fig1, use_container_width=True)

# =========================
# 2 - FALTAS
# =========================
if "faltas" in df.columns:

    st.subheader("📊 Média de faltas por risco")

    st.markdown("Mostra a média de faltas por nível de risco.")

    faltas = (
        df_filtrado.groupby("nivel_risco")["faltas"]
        .mean()
        .reindex(ordem_niveis)
        .reset_index()
    )

    fig2 = px.bar(
        faltas,
        x="nivel_risco",
        y="faltas",
        labels={
            "nivel_risco": "Nível de risco",
            "faltas": "Média de faltas"
        },
        color="nivel_risco",
        category_orders={"nivel_risco": ordem_niveis},
        color_discrete_map=cores_risco
    )

    st.plotly_chart(fig2, use_container_width=True)

# =========================
# 3 - ÁLCOOL
# =========================
if "consumo_alcool_fds" in df.columns:

    st.subheader("📊 Consumo de álcool por risco")

    alcool = (
        df_filtrado.groupby("nivel_risco")["consumo_alcool_fds"]
        .mean()
        .reindex(ordem_niveis)
        .reset_index()
    )

    fig3 = px.bar(
        alcool,
        x="nivel_risco",
        y="consumo_alcool_fds",
        labels={
            "nivel_risco": "Nível de risco",
            "consumo_alcool_fds": "Consumo médio"
        },
        color="nivel_risco",
        category_orders={"nivel_risco": ordem_niveis},
        color_discrete_map=cores_risco
    )

    st.plotly_chart(fig3, use_container_width=True)

# =========================
# 4 - APOIO FAMILIAR
# =========================
if "apoio_familia" in df.columns:

    st.subheader("📊 Apoio familiar por risco")

    apoio = (
        df_filtrado.groupby(["nivel_risco", "apoio_familia"])
        .size()
        .reset_index(name="quantidade")
    )

    apoio["apoio_familia"] = apoio["apoio_familia"].map({
        0: "Sem apoio",
        1: "Com apoio"
    })

    fig4 = px.bar(
        apoio,
        x="nivel_risco",
        y="quantidade",
        color="apoio_familia",
        barmode="group",
        labels={
            "nivel_risco": "Nível de risco",
            "quantidade": "Alunos",
            "apoio_familia": "Apoio familiar"
        },
        category_orders={"nivel_risco": ordem_niveis}
    )

    st.plotly_chart(fig4, use_container_width=True)

# =========================
# 5 - NOTAS
# =========================
st.subheader("📊 Distribuição das notas")

notas = df_filtrado.melt(
    id_vars="nivel_risco",
    value_vars=["nota_final"],
    var_name="tipo",
    value_name="nota"
)

fig5 = px.box(
    notas,
    x="nivel_risco",
    y="nota",
    labels={
        "nivel_risco": "Nível de risco",
        "nota": "Nota final"
    },
    category_orders={"nivel_risco": ordem_niveis}
)

st.plotly_chart(fig5, use_container_width=True)

# =========================
# 6 - GÊNERO POR RISCO
# =========================
st.subheader("📊 Risco por gênero")

genero_risco = (
    df_filtrado.groupby(["nivel_risco", "sexo"])
    .size()
    .reset_index(name="quantidade")
)

fig6 = px.bar(
    genero_risco,
    x="nivel_risco",
    y="quantidade",
    color="sexo",
    barmode="group",
    labels={
        "nivel_risco": "Nível de risco",
        "quantidade": "Alunos",
        "sexo": "Gênero"
    },
    category_orders={"nivel_risco": ordem_niveis},
    color_discrete_map={
        "M": "blue",
        "F": "red"
    }
)

st.plotly_chart(fig6, use_container_width=True)

# =========================
# 7 - IDADE E RISCO
# =========================
st.subheader("📊 Risco médio por idade")

st.markdown("Mostra a média de risco por idade dos alunos.")

taxa = (
    df_filtrado.groupby("idade")["prob_risco_%"]
    .mean()
    .reset_index()
)

fig7 = px.bar(
    taxa,
    x="idade",
    y="prob_risco_%",
    labels={
        "idade": "Idade do aluno",
        "prob_risco_%": "Risco médio (%)"
    }
)

st.plotly_chart(fig7, use_container_width=True)