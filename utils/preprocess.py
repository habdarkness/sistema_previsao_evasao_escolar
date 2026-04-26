import pandas as pd
from sklearn.preprocessing import LabelEncoder

def classificate(prob):
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


def load_data(caminho="data/student-mat-limpo.csv"):
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


def apply_encoders(
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

def get_features():
    return [
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