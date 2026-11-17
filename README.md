# Artificial Neural Networks (ANN) — Projetos em Python

Este repositório reúne **dois projetos académicos de redes neuronais artificiais**, desenvolvidos de raiz em Python, para explorar os fundamentos de uma **Multilayer Perceptron (MLP)** e a sua aplicação a problemas de **classificação binária**.

Em ambos os projetos, a rede é implementada sem bibliotecas de *machine learning* para o treino: os pesos, a propagação direta (*forward propagation*) e a retropropagação do erro (*backpropagation*) são programados manualmente. É utilizada uma camada escondida, a função de ativação **sigmoide** e a atualização dos pesos com uma taxa de aprendizagem configurável.

## Projetos

### 1. Flight Delay Prediction — Previsão de atrasos de voos

**Objetivo:** prever se um voo terá atraso (`Delayed = 1`) ou não (`Delayed = 0`), a partir das suas características.

- **Dados:** 1.000 voos simulados, equilibrados entre 500 com atraso e 500 sem atraso.
- **Entradas:** mês, dia da semana, hora prevista de partida, companhia aérea, origem, destino e distância.
- **Pré-processamento:** conversão de horas HHMM para minutos, normalização dos valores numéricos e codificação *one-hot* dos atributos categóricos.
- **Rede:** 54 entradas (calculadas a partir do dataset), 8 neurónios escondidos e 2 saídas na configuração predefinida; 50 épocas e taxa de aprendizagem de 0,1.
- **Avaliação:** *accuracy*, *precision*, *recall*, F1-score e matriz de confusão, com exportação de gráficos e tabela de resultados em PNG.

O script também inclui exemplos de aprendizagem das funções lógicas **AND, OR e XOR**, sendo o teste de AND executado automaticamente.

📁 **Pasta:** [`Flight-Delay-Prediction/`](Flight-Delay-Prediction/) — consulta o README dessa pasta para uma descrição técnica detalhada.

### 2. Student Depression Classification — Classificação de estudantes

**Objetivo:** investigar a classificação dos registos de estudantes relativamente à variável `Depression` (`Yes`/`No`) e comparar diferentes configurações da rede neuronal.

- **Dados:** 502 registos, com 252 casos `Yes` e 250 casos `No` no dataset fornecido.
- **Entradas previstas:** características como género, idade, hábitos alimentares, pressão académica, satisfação com os estudos, horas de estudo e stress financeiro.
- **Pré-processamento:** codificação *one-hot* de categorias e comparação de atributos numéricos normalizados e não normalizados.
- **Experiências:** 36 combinações de **4, 7 ou 11** neurónios escondidos, **5, 50 ou 100** épocas, dois tamanhos de treino/teste e normalização ativada/desativada.
- **Avaliação:** *accuracy* nos conjuntos de treino e teste, acompanhada de gráficos da evolução e da comparação de configurações.

**Limitação importante:** na versão atual, os índices de algumas colunas do CSV não correspondem ao pré-processamento implementado; além disso, a rede é criada com 18 entradas, embora a função de transformação produza 10. Assim, os resultados devem ser interpretados com cautela e o código necessita de correções antes de uma avaliação fiável. **Este projeto não é uma ferramenta de diagnóstico clínico.**

📁 **Pasta:** [`Student-Depression-Classification/`](Student-Depression-Classification/) — consulta o README dessa pasta para o funcionamento detalhado e as limitações identificadas.

## Estrutura do repositório

```text
Artificial-Neural-Networks/
├── README.md
├── Flight-Delay-Prediction/
│   ├── README.md
│   ├── nn_flights_alunos.py
│   └── flights_simulated_balanced_1000.csv
└── Student-Depression-Classification/
    ├── README.md
    ├── nn_students_alunos.py
    └── Depression Student Dataset.csv
```

## Como executar

É necessário ter **Python 3** e a biblioteca **Matplotlib**:

```bash
python -m pip install matplotlib
```

A partir da raiz do repositório, executa um dos projetos:

```bash
cd Flight-Delay-Prediction
python nn_flights_alunos.py
```

ou

```bash
cd Student-Depression-Classification
python nn_students_alunos.py
```

Mantém os ficheiros CSV nas respetivas pastas, pois os scripts utilizam caminhos relativos. O projeto de voos guarda imagens com as métricas; o de estudantes apresenta vários gráficos durante as experiências.

## Conceitos e tecnologias

**Python · Redes Neuronais Artificiais · MLP · Forward Propagation · Backpropagation · Classificação Binária · Pré-processamento de Dados · Matplotlib**

Estes projetos têm uma finalidade **académica e experimental**: demonstrar a implementação de redes neuronais e comparar os seus resultados, não disponibilizar sistemas de previsão prontos para produção.
