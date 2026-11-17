# Previsão de Atrasos de Voos com Redes Neuronais Artificiais

Projeto académico em Python que implementa **uma rede neuronal artificial do tipo Multilayer Perceptron (MLP), de raiz**, sem recorrer a bibliotecas de *machine learning* para a construção e o treino do modelo. O programa demonstra a aprendizagem de funções lógicas e aplica a mesma implementação à **classificação binária de voos: com atraso ou sem atraso**.

## Objetivos

- Implementar uma rede neuronal *feedforward* com uma camada escondida.
- Compreender a propagação direta (*forward propagation*) e a retropropagação do erro (*backpropagation*).
- Treinar a rede para aprender as funções lógicas **AND**, **OR** e **XOR**.
- Preparar dados de voos com atributos numéricos e categóricos.
- Classificar voos e avaliar o desempenho através de métricas e gráficos.

## Estrutura do projeto

```text
.
├── nn_flights_alunos.py                  # Implementação, treino, avaliação e visualização
├── flights_simulated_balanced_1000.csv   # Dataset de 1.000 voos simulados
└── README.md                             # Documentação do projeto
```

Após a execução, o programa também grava os ficheiros `teste_Gráfico.png` e `teste_Tabela.png` na diretoria de trabalho.

## Dataset

O ficheiro `flights_simulated_balanced_1000.csv` contém **1.000 registos simulados**, com distribuição equilibrada da variável-alvo: **500 voos sem atraso (`Delayed = 0`)** e **500 voos com atraso (`Delayed = 1`)**.

| Coluna | Descrição | Tratamento |
| --- | --- | --- |
| `Month` | Mês do voo (1–12) | Normalização Min-Max |
| `DayOfWeek` | Dia da semana (1–7) | Normalização Min-Max |
| `CRSDepTime` | Hora de partida programada no formato HHMM | Conversão para minutos após a meia-noite e normalização |
| `UniqueCarrier` | Código da companhia aérea | Codificação One-Hot |
| `Origin` | Aeroporto de origem | Codificação One-Hot |
| `Dest` | Aeroporto de destino | Codificação One-Hot |
| `Distance` | Distância do voo | Normalização Min-Max com os limites do dataset |
| `Delayed` | Classe real: `0` = sem atraso; `1` = com atraso | Variável-alvo, codificada em dois valores de saída |

Neste dataset existem **10 companhias aéreas**, **20 aeroportos de origem** e **20 aeroportos de destino** distintos. Assim, a representação de cada voo contém **54 entradas**: 4 atributos numéricos e 50 posições One-Hot (10 + 20 + 20).

O vetor de saída esperado é `[1, 0]` para **sem atraso** e `[0, 1]` para **com atraso**.

> **Nota:** Os dados são simulados. O projeto demonstra um procedimento de classificação e não constitui um serviço operacional de previsão de atrasos reais.

## Como funciona

### 1. Criação da rede neuronal

A função `make(nx, nz, ny)` cria uma rede totalmente ligada com:

- `nx` neurónios de entrada;
- `nz` neurónios numa camada escondida;
- `ny` neurónios na camada de saída.

Os pesos são inicializados aleatoriamente entre `-0.5` e `0.5`. A implementação inclui entradas constantes de valor `-1` para representar os limiares (*biases*) da camada escondida e da camada de saída.

A função de ativação utilizada é a **sigmoide**:

\[\sigma(x) = \frac{1}{1 + e^{-x}}\]

### 2. Propagação e aprendizagem

A aprendizagem é implementada pelas seguintes funções:

| Função | Responsabilidade |
| --- | --- |
| `forward()` | Calcula as ativações da camada escondida e das saídas a partir dos dados de entrada. |
| `error()` | Calcula os sinais de erro das saídas e da camada escondida. |
| `update()` | Atualiza os pesos utilizando os erros calculados e a taxa de aprendizagem `alpha`. |
| `iterate()` | Executa um passo de treino sobre um exemplo e apresenta os resultados, sendo utilizada nos testes das funções lógicas. |

O treino dos voos utiliza **atualização de pesos por exemplo** (*stochastic/online gradient descent*), com retropropagação do erro e uma taxa de aprendizagem definida por `alpha`.

### 3. Testes com funções lógicas

O código disponibiliza três funções de treino independentes:

- `train_and(epocas)` — aprende a função **AND**;
- `train_or(epocas)` — aprende a função **OR**;
- `train_xor(epocas)` — aprende a função **XOR**.

Cada função cria uma rede **2–2–1** e repete a aprendizagem dos quatro exemplos da respetiva tabela de verdade durante o número de épocas indicado. Quando o ficheiro é executado diretamente, é realizado o treino da função **AND** durante **1.000 épocas**, seguido da apresentação dos resultados para as quatro entradas possíveis. As funções OR e XOR estão implementadas, mas não são chamadas automaticamente.

### 4. Preparação dos dados de voos

A função `build_sets()` lê o ficheiro CSV, prepara os metadados e converte cada registo através de `translate()`:

1. Converte a hora de partida de HHMM para minutos desde a meia-noite.
2. Normaliza os quatro atributos numéricos quando `NORMALIZE = True`.
3. Converte companhia aérea, origem e destino para vetores One-Hot.
4. Converte a classe real no vetor de saída esperado.
5. Baralha os registos e divide-os em conjuntos de treino e de teste, sem sobreposição entre os dois conjuntos.

A função `get_input_size()` determina automaticamente o número de entradas necessário com base nas categorias identificadas no ficheiro.

### 5. Treino e classificação dos voos

`run_flights()` coordena a preparação dos dados e a execução de `train_flights()`.

Na configuração incluída no código, a rede utiliza:

| Parâmetro | Valor |
| --- | ---: |
| Entradas | 54 (calculadas automaticamente) |
| Neurónios escondidos | 8 |
| Saídas | 2 |
| Épocas de treino | 50 |
| Taxa de aprendizagem (`alpha`) | 0,1 |
| Exemplos de treino | 800 |
| Exemplos de teste | 200 |
| Normalização | Ativa |

Em cada época, os exemplos de treino são baralhados, a rede calcula uma previsão e os pesos são atualizados. A classe prevista corresponde ao índice do **maior valor das duas saídas**: índice 0 para voo sem atraso e índice 1 para voo com atraso.

### 6. Avaliação e visualização

A função `test_flights()` calcula:

- **Exatidão (*accuracy*):** proporção de previsões corretas.
- **Precisão (*precision*):** proporção de verdadeiros atrasos entre os voos previstos como atrasados.
- **Cobertura (*recall*):** proporção dos voos realmente atrasados que foram identificados como tal.
- **F1-score:** média harmónica entre precisão e cobertura.
- **Matriz de confusão:** verdadeiros positivos (VP), verdadeiros negativos (VN), falsos positivos (FP) e falsos negativos (FN), considerando *atrasado* como a classe positiva.

O programa regista a evolução da exatidão de treino, exatidão de teste, precisão, cobertura e F1-score ao longo das épocas. Gera ainda:

- `teste_Gráfico.png` — gráfico da evolução das métricas e resumo dos parâmetros da experiência;
- `teste_Tabela.png` — imagem com métricas e valores da matriz de confusão.

> Os resultados dependem da inicialização aleatória dos pesos e do baralhamento dos dados; o código não define uma semente aleatória fixa. Por esse motivo, diferentes execuções podem produzir métricas diferentes.

## Requisitos

- **Python 3**
- **Matplotlib** (única dependência externa importada pelo código)

As bibliotecas `random`, `math` e `csv` fazem parte da biblioteca padrão de Python.

Instalar a dependência externa:

```bash
python -m pip install matplotlib
```

## Como executar

1. Colocar `nn_flights_alunos.py` e `flights_simulated_balanced_1000.csv` na mesma pasta.
2. Abrir um terminal nessa pasta.
3. Instalar a dependência, caso ainda não esteja disponível.
4. Executar:

```bash
python nn_flights_alunos.py
```

A execução realiza, por esta ordem, o treino da função **AND** e o treino/teste da rede para **classificação de voos**. Os resultados são apresentados no terminal e as imagens são guardadas na diretoria de trabalho.

### Alterar os parâmetros

Os principais parâmetros podem ser ajustados no final de `nn_flights_alunos.py`:

```python
alpha = 0.1
NORMALIZE = True

# Configuração da classificação de voos
input_size, hidden_size, output_size = 18, 8, 2
epochs = 50
training_set_size = 800
test_set_size = 200
file = 'flights_simulated_balanced_1000.csv'
```

**Atenção:** Embora o valor inicial de `input_size` esteja definido como `18`, a função `run_flights()` substitui-o pelo número real de entradas determinado a partir do CSV (54 neste dataset). O número de saídas deve manter-se em **2** para a classificação binária implementada.

É possível experimentar diferentes números de neurónios escondidos, épocas, taxas de aprendizagem e tamanhos dos conjuntos de treino e teste. A soma dos dois conjuntos não pode ultrapassar o número de registos do dataset.

## Principais funções

| Função | Descrição |
| --- | --- |
| `make()` | Cria a rede e inicializa os pesos. |
| `sig()` | Aplica a função de ativação sigmoide. |
| `forward()` | Efetua a propagação direta. |
| `error()` | Calcula os erros para a retropropagação. |
| `update()` | Atualiza os pesos. |
| `train_and()`, `train_or()`, `train_xor()` | Treinam a rede para funções lógicas. |
| `_preparar_meta()` | Identifica categorias e limites de distância do CSV. |
| `build_sets()` | Constrói e separa os conjuntos de treino e teste. |
| `translate()` | Transforma um registo de voo num padrão para a rede. |
| `normaliza_valores()` | Normaliza atributos numéricos. |
| `converte_categ_numerico()` | Aplica a codificação One-Hot. |
| `run_flights()` | Coordena a experiência de classificação. |
| `train_flights()` | Treina a rede e produz o gráfico das métricas. |
| `test_flights()` | Calcula as métricas e a matriz de confusão. |
| `tabela_resultados()` | Exporta uma tabela de métricas em PNG. |
| `retranslate()` | Converte o vetor de saída numa descrição da classe prevista. |

## Limitações e possíveis melhorias

Esta implementação tem caráter pedagógico e existem alguns pontos a considerar:

1. **Inconsistência na tabela da matriz de confusão:** `test_flights()` envia os valores na ordem `[VP, FP, VN, FN]`, mas `tabela_resultados()` apresenta os índices 1 e 2 com os rótulos `VN` e `FP`, respetivamente. Assim, estes dois valores aparecem trocados **na tabela PNG**. Os cálculos de exatidão, precisão, cobertura e F1-score utilizam diretamente as contagens corretas.
2. **Utilização repetida do conjunto de teste:** o conjunto de teste é avaliado em todas as épocas. Para selecionar hiperparâmetros ou decidir quando terminar o treino, seria preferível acrescentar um conjunto de validação e reservar o teste para a avaliação final.
3. **Sem reprodutibilidade fixa:** não é definida uma semente aleatória para a inicialização e o baralhamento.
4. **Metadados calculados sobre todo o dataset:** as categorias e os valores mínimo/máximo da distância são determinados antes da divisão em treino e teste. Numa avaliação mais rigorosa, os parâmetros do pré-processamento devem ser ajustados apenas com o conjunto de treino.
5. **Sem persistência do modelo:** os pesos treinados não são guardados num ficheiro para reutilização posterior.
6. **Saídas gráficas sobrescritas:** as imagens têm nomes fixos; novas execuções substituem os ficheiros anteriores. Além disso, a tabela é gerada em todas as épocas, apesar de conservar apenas a última versão.
7. **Dataset simulado:** os resultados não permitem inferir diretamente o desempenho em dados reais de operações aéreas.

## Autores

- Luís Laia
- Rodrigo Lopes

---

*Projeto académico de implementação e avaliação de redes neuronais artificiais em Python.*
