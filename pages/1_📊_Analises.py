import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Análises", layout="wide")
st.title("📊 Dashboard de Análise de Risco Escolar")

# =========================
# ORDEM DOS NÍVEIS
# =========================
niveis = {
    "CRITICO - INTERVENÇÃO PEDAGOGICA PROFUNDA": "#E74C3C",
    "RUIM - PLANO DE RECUPERAÇÃO URGENTE": "#F39C12",
    "REGULAR - APOIO EXTRA OU TUTORIA": "#9ACD32",
    "BOM - MONITORIZAÇÃO OCASIONAL": "#90EE90",
    "EXCELENTE - MANTER AUTONOMIA": "#2ECC71",
}
niveis_keys = list(niveis.keys())

# =========================
# LABELS
# =========================

labels_padrao = {
    "nivel_risco": "Nível de Risco",
    "quantidade": "Quantidade de Alunos",
    "faltas": "Média de Faltas",
    "nota_final": "Nota Final",
    "sexo": "Gênero",
    "idade": "Idade",
    "prob_risco_%": "Risco Médio (%)",
    "apoio_familia": "Apoio Familiar",
    "consumo_alcool_fds": "Consumo Médio de Álcool (FDS)"
}

# =========================
# CARREGAR DADOS
# =========================
df = pd.read_csv("data/previsoes_alunos.csv")

df["nivel_risco"] = pd.Categorical(
    df["nivel_risco"],
    categories=niveis_keys,
    ordered=True
)

# =========================
# FILTROS
# =========================
st.sidebar.header("Filtros")

nivel = st.sidebar.multiselect("Nível de risco", niveis_keys, default=niveis_keys)

genero = st.sidebar.multiselect("Gênero", ["M", "F"], default=["M", "F"])

idade_min = st.sidebar.slider("Idade mínima", 15, 22, 15)
idade_max = st.sidebar.slider("Idade máxima", 15, 22, 22)

nota_min = st.sidebar.slider("Nota mínima", 0, 20, 0)
nota_max = st.sidebar.slider("Nota máxima", 0, 20, 20)

if "faltas" in df.columns:
    faltas_min = int(df["faltas"].min())
    faltas_max = int(df["faltas"].max())
    faltas_range = st.sidebar.slider(
        "Faixa de faltas",
        faltas_min,
        faltas_max,
        (faltas_min, faltas_max)
    )
    faltas_min_sel, faltas_max_sel = faltas_range
else:
    faltas_min_sel, faltas_max_sel = 0, 999

# =========================
# DATAFRAME FILTRADO
# =========================
df_filtrado = df[
    (df["nivel_risco"].isin(nivel)) &
    (df["nota_final"] >= nota_min) &
    (df["nota_final"] <= nota_max) &
    (df["idade"] >= idade_min) &
    (df["idade"] <= idade_max) &
    (df["sexo"].isin(genero)) &
    (df["faltas"] >= faltas_min_sel) &
    (df["faltas"] <= faltas_max_sel)
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
st.markdown("**Objetivo:** Apresentar a quantidade de alunos em cada categoria de risco escolar para visualizar o panorama geral da instituição.")

contagem = (
    df_filtrado["nivel_risco"]
    .value_counts()
    .reindex(niveis_keys)
    .reset_index()
)

contagem.columns = ["nivel_risco", "quantidade"]

fig1 = px.bar(
    contagem,
    x="nivel_risco",
    y="quantidade",
    color="nivel_risco",
    labels=labels_padrao,
    category_orders={"nivel_risco": niveis_keys},
    color_discrete_map=niveis
)

st.plotly_chart(fig1, use_container_width=True)

# =========================
# 2 - FALTAS
# =========================
if "faltas" in df.columns:
    st.subheader("📊 Média de faltas por risco")
    st.markdown("**Objetivo:** Mostrar a relação entre frequência escolar e nível de risco, identificando se faltas elevadas estão associadas a maior vulnerabilidade.")

    faltas = (
        df_filtrado.groupby("nivel_risco")["faltas"]
        .mean()
        .reindex(niveis_keys)
        .reset_index()
    )

    fig2 = px.bar(
        faltas,
        x="nivel_risco",
        y="faltas",
        color="nivel_risco",
        labels=labels_padrao,
        category_orders={"nivel_risco": niveis_keys},
        color_discrete_map=niveis
    )

    st.plotly_chart(fig2, use_container_width=True)

# =========================
# 3 - ÁLCOOL
# =========================
if "consumo_alcool_fds" in df.columns:
    st.subheader("📊 Consumo de álcool por risco")
    st.markdown("**Objetivo:** Avaliar possíveis impactos de hábitos externos, como consumo de álcool, sobre o desempenho e risco escolar.")

    alcool = (
        df_filtrado.groupby("nivel_risco")["consumo_alcool_fds"]
        .mean()
        .reindex(niveis_keys)
        .reset_index()
    )

    fig3 = px.bar(
        alcool,
        x="nivel_risco",
        y="consumo_alcool_fds",
        color="nivel_risco",
        labels=labels_padrao,
        category_orders={"nivel_risco": niveis_keys},
        color_discrete_map=niveis
    )

    st.plotly_chart(fig3, use_container_width=True)

# =========================
# 4 - APOIO FAMILIAR
# =========================
if "apoio_familia" in df.columns:
    st.subheader("📊 Apoio familiar por risco")
    st.markdown("**Objetivo:** Analisar a influência do suporte familiar na distribuição dos níveis de risco dos alunos.")

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
        labels=labels_padrao,
        barmode="group",
        category_orders={"nivel_risco": niveis_keys}
    )

    st.plotly_chart(fig4, use_container_width=True)

# =========================
# 5 - NOTAS
# =========================
st.subheader("📊 Distribuição das notas")
st.markdown("**Objetivo:** Comparar o desempenho acadêmico entre diferentes grupos de risco por meio da variação das notas finais.")

fig5 = px.box(
    df_filtrado,
    x="nivel_risco",
    y="nota_final",
    labels=labels_padrao,
    category_orders={"nivel_risco": niveis_keys}
)

st.plotly_chart(fig5, use_container_width=True)

# =========================
# 6 - GÊNERO
# =========================
st.subheader("📊 Risco por gênero")
st.markdown("**Objetivo:** Identificar diferenças na distribuição de risco escolar entre gêneros para análise demográfica.")

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
    labels=labels_padrao,
    barmode="group",
    category_orders={"nivel_risco": niveis_keys},
    color_discrete_map={"M": "blue", "F": "red"}
)

st.plotly_chart(fig6, use_container_width=True)

# =========================
# 7 - IDADE
# =========================
st.subheader("📊 Risco médio por idade")
st.markdown("**Objetivo:** Detectar faixas etárias com maior propensão ao risco escolar para orientar estratégias preventivas.")

taxa = (
    df_filtrado.groupby("idade")["prob_risco_%"]
    .mean()
    .reset_index()
)

fig7 = px.bar(
    taxa,
    x="idade",
    y="prob_risco_%",
    labels=labels_padrao,
)

st.plotly_chart(fig7, use_container_width=True)