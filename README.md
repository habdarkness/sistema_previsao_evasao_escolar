# 📊 Sistema Inteligente de Índice de Desempenho Escolar

> Aplicação inteligente para previsão de risco de baixo desempenho escolar, desenvolvida como parte de um PBL do curso de **Engenharia de Software da UNDB**.

---

## 📌 Sobre o Projeto

Este sistema utiliza **Inteligência Artificial** para classificar o risco de baixo desempenho escolar de alunos com base em dados históricos acadêmicos, sociais e comportamentais. A interface foi construída com **Streamlit**, proporcionando uma experiência interativa e acessível para educadores e gestores escolares.

---

## 🚀 Funcionalidades

- 📈 **Análises Exploratórias** — Gráficos interativos sobre notas, tempo de estudo, consumo de álcool, faltas e outros indicadores relevantes.
- 🔮 **Previsão de Risco** — Classificação binária (**baixo risco / alto risco**) utilizando uma rede neural MLP treinada.
- 🖥️ **Dashboard Interativo** — Filtros dinâmicos para explorar e segmentar os dados por diferentes variáveis.
- 💾 **Modelos Salvos** — Carregamento automático de encoders, scaler e rede neural previamente treinados.

---

## 📂 Estrutura do Projeto

```
📦 sistema-desempenho-escolar/
├── 📁 data/
│   └── student_data.csv          # Base de dados dos alunos
├── 📁 models/
│   ├── mlp_model.pkl             # Rede neural MLP treinada
│   ├── scaler.pkl                # Scaler para normalização
│   └── encoders.pkl              # Encoders para variáveis categóricas
├── 📁 pages/
│   ├── 01_dashboard.py           # Dashboard com filtros dinâmicos
│   ├── 02_analise_exploratoria.py # Gráficos e análises
│   └── 03_previsao.py            # Módulo de previsão de risco
├── app.py                        # Ponto de entrada da aplicação
├── requirements.txt              # Dependências do projeto
└── README.md
```

---

## 🛠️ Tecnologias Utilizadas

| Tecnologia | Finalidade |
|---|---|
| Python 3.10+ | Linguagem principal |
| Streamlit | Interface web interativa |
| Scikit-learn | Rede neural MLP e pré-processamento |
| Pandas | Manipulação e análise de dados |
| Plotly / Matplotlib | Visualizações gráficas |
| Joblib | Serialização dos modelos treinados |

---

## ⚙️ Como Executar

### 1. Clone o repositório

```bash
git clone https://github.com/seu-usuario/sistema-desempenho-escolar.git
cd sistema-desempenho-escolar
```

### 2. Crie e ative um ambiente virtual

```bash
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows
```

### 3. Instale as dependências

```bash
pip install -r requirements.txt
```

### 4. Execute a aplicação

```bash
streamlit run app.py
```

A aplicação estará disponível em `http://localhost:8501`.

---

## 🤖 Modelo de Machine Learning

O modelo de previsão é uma **Rede Neural MLP (Multi-Layer Perceptron)** treinada com dados acadêmicos, sociais e comportamentais dos alunos. As principais variáveis utilizadas incluem:

- Notas anteriores (G1, G2)
- Tempo de estudo semanal
- Número de reprovações anteriores
- Frequência escolar (faltas)
- Consumo de álcool (dias de semana e fim de semana)
- Suporte familiar e escolar
- Atividades extracurriculares

O pipeline de pré-processamento inclui normalização via `StandardScaler` e codificação de variáveis categóricas com `LabelEncoder`, ambos persistidos para uso em produção.

---

## 📊 Exemplos de Visualizações

- Distribuição de notas finais por turma
- Correlação entre tempo de estudo e desempenho
- Impacto do consumo de álcool nas notas
- Taxa de faltas por nível de risco
- Mapa de calor de correlações entre variáveis

---

## 👥 Equipe

Desenvolvido por alunos do curso de **Engenharia de Software — UNDB** como parte de um projeto de Problem-Based Learning (PBL).

---

## 📄 Licença

Este projeto está sob a licença [MIT](LICENSE).
