import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# IMPORTAÇÕES
# Pandas - manipulação de dados
# Numpy - operações matematicas com arrays
# sklearn - ferramentas de machine learning:
#   onde: train_test_split - divide os dados em treino
#         StandardScaler - normalização dos dados
#         LabelEncoder - transformar texto em numero
#         classification_report - metricas de avaliação
#         confusion_matrix - tabela de acertos e erros
#         f1_score - metrica para testar thresholds
# tensorflow/keras - biblioteca para construir a rede neural


#carregando o dataset
df = pd.read_csv("data/student-mat-limpo.csv")

#Se a nota final for menor que 10, o aluno está em risco (1), caso contrario, (0). O operador < retorna True/False e o astype(int) converte 1/0
df["risco"] = (df["nota_final"] < 10).astype(int)



#Mostra quantos alunos estão dentro do baixo e alto risco
print(f"\nDistribuição do risco:")
print(f"  Baixo risco (nota >= 10): {(df['risco'] == 0).sum()} alunos")
print(f"  Alto risco  (nota <  10): {(df['risco'] == 1).sum()} alunos")

#As colunas selecionadas contem apenas yes/no são transformadas diretamente em 1 e 0 (Variaveis binarias)
colunas_binarias = [
    "apoio_escola", "apoio_familia", "aulas_pagas",
    "atividades_extracurriculares", "frequentou_creche",
    "deseja_ensino_superior", "acesso_internet", "relacionamento_romantico",
]

#para cada coluna substitui o yes=1 e outro valor por 0
for col in colunas_binarias:
    df[col] = (df[col] == "yes").astype(int)


#Codifica colunas estrategicas com LabelEncoder
#Colunas com texto que representam categorias sao convertidas para numeros inteiros pelo LabelEncoder
#Guarda essas informações em um dicionario

colunas_categoricas = [
    "escola", "sexo", "tipo_residencia", "tamanho_familia",
    "situacao_pais", "trabalho_mae", "trabalho_pai",
    "motivo_escola", "responsavel",
]
encoders = {}  # dicionário que guarda um encoder por coluna
for col in colunas_categoricas:
    le = LabelEncoder()
    # fit_transform aprende os valores únicos e já transforma a coluna
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le  # salva o encoder para usar depois na previsão


#Define as variaveis de entrada do modelo
#Exclui-se nota_final pois ela é a variavel alvo e o modelo poderia "trapacear"


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

#Separa as features (X) da variável alvo (y)
X = df[features].values  #entrada: cada linha é um aluno
y = df["risco"].values   #saída: 0 ou 1 para cada aluno

# Dividir: 70% treino, 15% validação, 15% teste
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

#Normalizando os dados com o Standard para transformart todos os valores para uma escala parecida
#fit apenas no treino, transform nos demais (evita data leakage)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)

print(f"\nTreino: {X_train.shape[0]} | Validação: {X_val.shape[0]} | Teste: {X_test.shape[0]}")



#ReLU: retorna 0 se o valor for negativo, ou o proprio valor se positivo. Permite que a rede aprenda padrões não lineares nos dados
#Sigmoid: tranforma qualquer numero em um valor entre 0 e 1
modelo = keras.Sequential([
    layers.Input(shape=(X_train.shape[1],)), #camada de entrada

    layers.Dense(64, activation="relu"), #camada oculta #1
    layers.Dropout(0.3), #desliga aleatoriamente 30% dos neuronios, evitando overffiting

    layers.Dense(32, activation="relu"), #camada oculta #2, refinando os padroes aprendidos
    layers.Dropout(0.2), #dropout menor visto que a rede está menor

    layers.Dense(16, activation="relu"), #camada oculta #3, ultima camada de extração de padrões

    layers.Dense(1, activation="sigmoid"), #camada de saida: transforma qualquer valor em um numero entre 0 e 1
])


#compilar o modelo
#otimizador adam: ajusta dos pesos da rede durante o treino
#learning rate: tamanho do passo de ajuste a cada iteração (se muito alto, o modelo nao convergem. Se muito baixo, modelo aprende devagar)
#loss: função de erro para a classificação binaria (0 ou 1) medindo o quao erradas estao as probabilidades previstas
#metrics: exibe a acuracia do modelo
modelo.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)
#exibe um resumo da arquitetura
modelo.summary()


#early_stopping: para o treino automaticamente se o modelo parar de melhorar
#"val_loss": observa o erro na validação a cada epoca
#pacience: aguarda 15 epocas psem melhora antes de parar
#restore_best_weights: ao parar restaura os pesos da melhor epoca dos dados evitando que o modelo "decore" os dados

early_stop = keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=15, restore_best_weights=True, verbose=1
)

#modelo fit executa o treinamento usando os dados de treino, com o numero maxio de epocas,
#quantos alunos serão processados por vez antes de ajustar os pesos, dados de validação para monitorar o desempenho
#e lista de funções extras chamadas durante o treino.
historico = modelo.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[early_stop],
    class_weight={0: 1, 1: 2},
    verbose=1,
)

#probabilidades brutas de cada aluno do conjunto de teste
#o modelo usando flatten() transforma um array para uma lista simples de probabilidades entre 0 e 1

y_prob = modelo.predict(X_test).flatten()

#aplica o threshold de 0.2 para converter probabilidades em classes (0 ou 1)
y_pred = (y_prob >= 0.2).astype(int)

#usa o classification_report para exibir as metricas de desempenho como:
#precision: quantidade de "risco"
#recall: quantos o modelo de fato encontrou dentro do "risco"
#f1-score: media entre precision e recall
#support: total de alunos nessa classe no conjunto de teste
print("\n" + "=" * 50)
print("RESULTADOS DENTRO DO CONJUNTO DE TESTE")
print("=" * 50)
print(classification_report(y_test, y_pred, target_names=["Baixo Risco", "Alto Risco"]))
print("Matriz de Confusão:")
print(confusion_matrix(y_test, y_pred))


#reutiliza o mesmo dataset separado acima
X_todos = df[features].values

#normaliza usando o transform para manter a mesma escala
X_todos_norm = scaler.transform(X_todos)

#gera a probabilidade de risco para cada aluno
probabilidades = modelo.predict(X_todos_norm, verbose=0).flatten()

#cria um DataFrame com os resultados para facilitar a visualização
#cria um ID sequencial para cada aluno
#probabilidade em percentual
#1 = sem risco, 0=com risco
#usa valor real para comparação
resultados = pd.DataFrame({
    "aluno_id": range(1, len(df) + 1),

    "nota_periodo_1": df["nota_periodo_1"].values,
    "nota_periodo_2": df["nota_periodo_2"].values,
    "nota_final": df["nota_final"].values,

    #
    "faltas": df["faltas"].values,
    "tempo_estudo": df["tempo_estudo"].values,
    "consumo_alcool_fds": df["consumo_alcool_fds"].values,
    "apoio_familia": df["apoio_familia"].values,
    "sexo": df["sexo"].map({
        0: "F",
        1: "M"
    }),
    "idade": df["idade"],

    "prob_risco_%": (probabilidades * 100).round(1),
    "risco_previsto": (probabilidades >= 0.4).astype(int),
    "risco_real": df["risco"].values,
    "reprovação": df["reprovacoes_anteriores"].values,
})
#classifica o nível de risco com base na probabilidade
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
        # Só vai cair aqui se a probabilidade for maior que 100 (ou houver algum erro nos dados)
        return "ALTO"

resultados["nivel_risco"] = resultados["prob_risco_%"].apply(classificar_nivel)

#exibe um resumo geral
print("\n" + "=" * 55)
print("PREVISÃO DE RISCO")
print("=" * 55)
print(f"Total de alunos: {len(resultados)}")
print(f"Em risco:  {resultados['risco_previsto'].sum()} alunos")
print(f"Sem risco: {(resultados['risco_previsto'] == 0).sum()} alunos")

print(f"\nDistribuição por nível:")
print(resultados["nivel_risco"].value_counts().to_string())

#exibe apenas os alunos identificados como alto risco
print("\n" + "-" * 55)
print("Alunos com alto risco")
print("-" * 55)
alto_risco = resultados[resultados["nivel_risco"] == "CRITICO - INTERVENÇÃO PEDAGOGICA PROFUNDA"].sort_values(
    "prob_risco_%", ascending=False
)
print(alto_risco[["aluno_id", "nota_periodo_1", "nota_periodo_2",
                   "nota_final", "prob_risco_%","reprovação"]].to_string(index=False))

#salva o resultado completo em CSV para a prefeitura para melhor analise
resultados.to_csv("data/previsoes_alunos.csv", index=False)
print("\n Arquivo 'previsoes_alunos.csv' salvo")



import joblib

# salvar scaler e encoders
joblib.dump(scaler, "model/scaler.pkl")
joblib.dump(encoders, "model/encoders.pkl")

# salvar modelo
modelo.save("model/modelo.h5")