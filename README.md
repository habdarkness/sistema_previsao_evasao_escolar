```markdown
# 📊 Sistema Inteligente de Índice de Desempenho Escolar

Este projeto é uma aplicação em **Streamlit** que utiliza **Inteligência Artificial** para prever o risco de baixo desempenho escolar de alunos, com base em dados históricos acadêmicos, sociais e comportamentais.  
A solução foi desenvolvida como parte de um PBL do curso de Engenharia de Software da **UNDB**.

---

## 🚀 Funcionalidades

- 📈 **Análises exploratórias**: gráficos interativos sobre notas, tempo de estudo, consumo de álcool, faltas etc.
- 🔮 **Previsão de risco**: classificação binária (baixo risco / alto risco) usando rede neural MLP.
- 🖥️ **Dashboard interativo**: filtros dinâmicos para explorar os dados.
- 💾 **Modelos salvos**: carregamento de encoders, scaler e rede neural treinada.

---

## 📂 Estrutura do Projeto

```
SISTEMA_PREVISAO_EVASAO/
│── app.py                # Arquivo principal da aplicação Streamlit
│── requirements.txt       # Dependências do projeto
│── README.md              # Documentação do projeto
│── runtime.txt            # Configuração de ambiente
│── data/                  # Bases de dados e análises
│   ├── student-mat.csv
│   ├── student-mat-limpo.csv
│   ├── metricas_modelo.csv
│   └── previsoes_alunos.csv
│── model/                 # Modelos e artefatos
│   ├── modelo.keras
│   ├── modelo.py
│   ├── encoders.pkl
│   └── scaler.pkl
│── pages/                 # Páginas da aplicação
│   ├── 1_📊_Analises.py
│   └── 2_🔮_Previsao.py
│── utils/                 # Funções auxiliares
│   └── preprocess.py
```

---

## 🛠️ Tecnologias Utilizadas

- **Python 3.13**
- **Streamlit** → Interface web interativa
- **Plotly** → Gráficos dinâmicos
- **Pandas / NumPy** → Manipulação e análise de dados
- **Scikit-learn** → Pré-processamento e métricas
- **TensorFlow / Keras** → Rede neural MLP
- **Joblib** → Salvamento e carregamento de modelos

---

## 📦 Instalação

Clone o repositório e instale as dependências:

```bash
git clone https://github.com/habdarkness/sistema_previsao_evasao_escolar.git
cd SISTEMA_PREVISAO_EVASAO_ESCOLAR
pip install -r requirements.txt
```

---

## ▶️ Executando a Aplicação

Após instalar as dependências, execute:

```bash
streamlit run app.py
```

---

## 📊 Dataset Utilizado

O projeto utiliza o dataset **Student Alcohol Consumption** (Kaggle), especificamente o arquivo `student-mat.csv`, que contém:

- Dados acadêmicos (notas, faltas, reprovações)
- Dados familiares
- Dados sociais e comportamentais
- Informações de saúde e hábitos

---

## 👥 Equipe

Projeto desenvolvido por estudantes do curso de **Engenharia de Software da UNDB**, sob orientação da disciplina de **Inteligência Artificial**.

---

## 📜 Licença

Este projeto é de uso acadêmico e está sob licença MIT.
```