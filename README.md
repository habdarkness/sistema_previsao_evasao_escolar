# Sistema Inteligente de Índice de Desempenho Escolar

Aplicação para previsão de risco de baixo desempenho escolar, desenvolvida como parte de um PBL do curso de **Engenharia de Software da UNDB**.

---

## Sobre o Projeto

Este sistema utiliza Inteligência Artificial para classificar o risco de baixo desempenho de alunos com base em dados históricos acadêmicos, sociais e comportamentais. A interface foi construída com **Streamlit** e os dados utilizados são provenientes de duas escolas secundárias de Portugal (ano letivo 2005–2006).

**Fonte dos dados:** [Kaggle — Student Alcohol Consumption](https://www.kaggle.com/datasets/uciml/student-alcohol-consumption/data)  
**Arquivo:** `student-mat.csv` (disciplina de Matemática) — 395 alunos × 33 colunas

---

## Funcionalidades

- **Análises Exploratórias** — Gráficos interativos sobre notas, tempo de estudo, consumo de álcool, faltas e outros indicadores.
- **Previsão de Risco** — Classificação binária (baixo risco / alto risco) usando rede neural MLP.
- **Dashboard Interativo** — Filtros dinâmicos para explorar e segmentar os dados.
- **Modelos Salvos** — Carregamento automático de encoders, scaler e rede neural treinada.

---

## Dataset

Os dados combinam informações escolares (notas, faltas, reprovações) com contexto familiar e comportamental (consumo de álcool, tempo de estudo, acesso à internet).

### Principais colunas

| Coluna | Descrição |
|--------|-----------|
| `escola` | Escola do aluno: GP ou MS |
| `sexo` | F = Feminino, M = Masculino |
| `idade` | Idade do aluno (15–22 anos) |
| `tempo_estudo` | Horas de estudo por semana: 1 = <2h … 4 = >10h |
| `reprovacoes_anteriores` | Número de reprovações anteriores (0–4) |
| `consumo_alcool_semana` | Consumo de álcool nos dias úteis: 1 = muito baixo … 5 = muito alto |
| `consumo_alcool_fds` | Consumo de álcool nos fins de semana: 1 = muito baixo … 5 = muito alto |
| `faltas` | Número de faltas no ano (0–93) |
| `nota_periodo_1` | Nota do 1º período (0–20) |
| `nota_periodo_2` | Nota do 2º período (0–20) |
| `nota_final` | **Nota final (0–20)** — variável alvo |

> A lista completa de colunas está disponível no notebook `analise_exploratoria.ipynb`.

---

## Modelo de Machine Learning

O modelo é uma **Rede Neural MLP (Multi-Layer Perceptron)** treinada com variáveis acadêmicas, sociais e comportamentais. O pipeline inclui normalização via `StandardScaler` e codificação de variáveis categóricas com `LabelEncoder`, ambos persistidos para uso em produção.

---

## Estrutura do Projeto

```
├── .devcontainer/
│   └── devcontainer.json           # Configuração do Dev Container
├── .vscode/
│   └── settings.json               # Configurações do editor
├── data/
│   ├── analise_exploratoria.ipynb  # Notebook de análise exploratória
│   ├── matriz_confusao.csv         # Resultado da matriz de confusão
│   ├── metricas_modelo.csv         # Métricas de avaliação do modelo
│   ├── previsoes_alunos.csv        # Previsões geradas pelo modelo
│   ├── resumo_modelo.csv           # Resumo de desempenho do modelo
│   ├── student-mat-limpo.csv       # Dados tratados (colunas em português)
│   └── student-mat.csv             # Dados originais
├── model/
│   ├── encoders.pkl                # Encoders para variáveis categóricas
│   ├── modelo.keras                # Rede neural treinada (Keras)
│   ├── modelo.py                   # Script de treinamento do modelo
│   └── scaler.pkl                  # Scaler para normalização
├── pages/
│   ├── 1_Analises.py               # Página de análises exploratórias
│   └── 2_Previsao.py               # Página de previsão de risco
├── utils/
│   ├── __init__.py
│   └── preprocess.py               # Funções de pré-processamento
├── .gitignore
├── README.md
├── app.py                          # Ponto de entrada da aplicação
├── requirements.txt
└── runtime.txt                     # Versão do Python para deploy
```

---

## Tecnologias

| Tecnologia | Finalidade |
|---|---|
| Python 3.10+ | Linguagem principal |
| Streamlit | Interface web interativa |
| Scikit-learn | Rede neural MLP e pré-processamento |
| Pandas | Manipulação de dados |
| Plotly / Matplotlib | Visualizações gráficas |
| Joblib | Serialização dos modelos |

---

## Como Executar

```bash
# Clone o repositório
git clone https://github.com/seu-usuario/sistema-desempenho-escolar.git
cd sistema-desempenho-escolar

# Crie e ative o ambiente virtual
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows

# Instale as dependências e execute
pip install -r requirements.txt
streamlit run app.py
```

A aplicação estará disponível em `http://localhost:8501`.

---

## Equipe

Desenvolvido por alunos do curso de Engenharia de Software — UNDB como parte de um projeto de Problem-Based Learning (PBL).

## Licença

Este projeto está sob a licença [MIT](LICENSE).
