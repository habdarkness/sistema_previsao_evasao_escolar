import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from utils.preprocess import classificate, load_data, encode, get_features

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
df = load_data()
df, encoders = encode(df)

#Mostra quantos alunos estão dentro do baixo e alto risco
print(f"\nDistribuição do risco:")
print(f"  Baixo risco (nota >= 10): {(df['risco'] == 0).sum()} alunos")
print(f"  Alto risco  (nota <  10): {(df['risco'] == 1).sum()} alunos")

#Define as variaveis de entrada do modelo
#Exclui-se nota_final pois ela é a variavel alvo e o modelo poderia "trapacear"
features = get_features()

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

#aplica o threshold de 0.4 para converter probabilidades em classes (0 ou 1)
TRESHOLD = 0.4
y_pred = (y_prob >= TRESHOLD).astype(int)

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
    "sexo": encoders["sexo"].inverse_transform(df["sexo"]),
    "idade": df["idade"],

    "prob_risco_%": (probabilidades * 100).round(1),
    "risco_previsto": (probabilidades >= TRESHOLD).astype(int),
    "risco_real": df["risco"].values,
    "reprovação": df["reprovacoes_anteriores"].values,
})
#classifica o nível de risco com base na probabilidade

resultados["nivel_risco"] = resultados["prob_risco_%"].apply(classificate)

#salva as metricas
from sklearn.metrics import classification_report
report = classification_report(
    y_test,
    y_pred,
    target_names=["Baixo Risco", "Alto Risco"],
    output_dict=True
)
df_report = pd.DataFrame(report).transpose()
df_report.to_csv("data/metricas_modelo.csv")
print("\nMetricas salvas em 'data/metricas_modelo.csv'")

#salva a matriz de confusão
cm = confusion_matrix(y_test, y_pred)
df_cm = pd.DataFrame(
    cm,
    index=["Real Baixo", "Real Alto"],
    columns=["Previsto Baixo", "Previsto Alto"]
)
df_cm.to_csv("data/matriz_confusao.csv")
print("Matriz de confusão salva em 'data/matriz_confusao.csv'")

#Salva o resumo
resumo = pd.DataFrame({
    "total_alunos": [len(resultados)],
    "alunos_risco": [resultados["risco_previsto"].sum()],
    "alunos_sem_risco": [(resultados["risco_previsto"] == 0).sum()]
})

resumo.to_csv("data/resumo_modelo.csv", index=False)
print("Resumo salvo em 'data/resumo_modelo.csv'")

# salvar scaler e encoders e o modelo
import joblib
joblib.dump(scaler, "model/scaler.pkl")
joblib.dump(encoders, "model/encoders.pkl")
modelo.save("model/modelo.h5")
print("Modelo salvo em 'data/'")

#salva o resultado completo em CSV para a prefeitura para melhor analise
resultados.to_csv("data/previsoes_alunos.csv", index=False)
print("Resultado salvo em 'previsoes_alunos.csv' salvo")