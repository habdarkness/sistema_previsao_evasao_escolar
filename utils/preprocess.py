import pandas as pd
from sklearn.preprocessing import LabelEncoder

def classificar_nivel(prob):
    if prob <= 20:
        return "EXCELENTE - MANTER AUTONOMIA"
    elif prob <= 40:
        return "BOM - MONITORIZAÇÃO OCASIONAL"
    elif prob <= 60:
        return "REGULAR - APOIO EXTRA OU TUTORIA"
    elif prob <= 80:
        return "RUIM - PLANO DE RECUPERAÇÃO URGENTE"
    elif prob <= 100:
        return "CRITICO - INTERVENÇÃO PEDAGOGICA PROFUNDA"
    else:
        return "ALTO"


def carregar_dados(caminho="data/student-mat-limpo.csv"):
    df = pd.read_csv(caminho)

    # variável alvo
    df["risco"] = (df["nota_final"] < 10).astype(int)

    # binárias
    colunas_binarias = [
        "apoio_escola", "apoio_familia", "aulas_pagas",
        "atividades_extracurriculares", "frequentou_creche",
        "deseja_ensino_superior", "acesso_internet", "relacionamento_romantico",
    ]

    for col in colunas_binarias:
        df[col] = (df[col] == "yes").astype(int)

    return df


def aplicar_encoders(
    df,
    colunas_categoricas=[
        "escola", "sexo", "tipo_residencia", "tamanho_familia",
        "situacao_pais", "trabalho_mae", "trabalho_pai",
        "motivo_escola", "responsavel",
    ]
):
    encoders = {}

    for col in colunas_categoricas:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    return df, encoders