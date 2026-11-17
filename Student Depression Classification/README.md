# Classificação de Depressão em Estudantes com Redes Neuronais

Projeto académico em **Python** que implementa, de raiz, uma **rede neuronal artificial do tipo Multilayer Perceptron (MLP)**, com uma camada escondida, para explorar a classificação de registos de estudantes relativamente à variável **`Depression`** (`Yes`/`No`). O programa inclui ainda exemplos de aprendizagem das funções lógicas **AND**, **OR** e **XOR**.

> **Nota importante:** este projeto é uma experiência de aprendizagem automática com dados tabulares. **Não constitui um instrumento clínico de diagnóstico ou avaliação de risco de depressão.** A implementação fornecida apresenta incompatibilidades entre a ordem das colunas do CSV e o pré-processamento; por isso, os resultados devem ser interpretados com especial cautela até essas incompatibilidades serem corrigidas.

## Conteúdo do repositório

```text
.
├── nn_students_alunos.py
├── Depression Student Dataset.csv
└── README.md
```

- **`nn_students_alunos.py`** — definição, treino, avaliação e comparação das redes neuronais.
- **`Depression Student Dataset.csv`** — conjunto de dados utilizado nas experiências.
- **`README.md`** — documentação e instruções de execução.

## Objetivo

Investigar a influência de diferentes configurações de uma rede neuronal no desempenho de uma tarefa de **classificação binária**, incluindo:

- número de neurónios na camada escondida;
- número de épocas de treino;
- dimensão dos conjuntos de treino e teste;
- utilização ou não de normalização dos atributos numéricos.

O código utiliza uma MLP construída sem bibliotecas de aprendizagem automática, permitindo observar explicitamente a propagação direta (*forward propagation*), o cálculo do erro por *backpropagation* e a atualização dos pesos.

## Dataset

O ficheiro fornecido contém **502 registos de estudantes e 11 colunas**, incluindo a variável de saída. A variável `Depression` está distribuída em **252 registos `Yes`** e **250 registos `No`**.

| Coluna | Descrição |
| --- | --- |
| `Gender` | Género registado |
| `Age` | Idade |
| `Academic Pressure` | Pressão académica |
| `Study Satisfaction` | Satisfação com os estudos |
| `Sleep Duration` | Duração do sono |
| `Dietary Habits` | Hábitos alimentares |
| `Have you ever had suicidal thoughts ?` | Resposta à pergunta sobre pensamentos suicidas prévios |
| `Study Hours` | Horas de estudo |
| `Financial Stress` | Stress financeiro |
| `Family History of Mental Illness` | Antecedentes familiares de doença mental |
| `Depression` | Classe a prever: `Yes` ou `No` |

**Atenção:** a presença de uma coluna no dataset não significa que o código a utilize. A função `translate()` seleciona apenas alguns índices e, na versão fornecida, parte dessa seleção não corresponde aos nomes das colunas (ver [Limitações conhecidas](#limitações-conhecidas)).

## Arquitetura da rede neuronal

A função `make(nx, nz, ny)` cria uma rede *feedforward* com:

1. **Camada de entrada:** recebe o vetor numérico de características do estudante.
2. **Camada escondida:** contém um número configurável de neurónios.
3. **Camada de saída:** contém **2 neurónios**, correspondentes às classes `Yes` e `No`.

São usados pesos iniciais aleatórios no intervalo `[-0.5, 0.5]`, função de ativação **sigmoide** e unidades adicionais com valor fixo `-1` para implementar os limiares (*bias*). A aprendizagem ajusta os pesos através de *backpropagation*, com **taxa de aprendizagem `alpha = 0.2`**.

### Codificação da saída

| Classe do CSV | Vetor pretendido |
| --- | --- |
| `Yes` | `[1, 0]` |
| `No` | `[0, 1]` |

A função `retranslate()` escolhe a classe associada à saída de maior valor. Em caso de empate, escolhe `[0, 1]` (`No`). Os valores da camada de saída são ativações sigmoides independentes, **não probabilidades calibradas**.

## Pré-processamento implementado

O código pretende converter atributos categóricos em vetores **one-hot** e, quando a opção está ativa, normalizar atributos numéricos para escalas predefinidas:

| Característica pretendida | Transformação prevista |
| --- | --- |
| Género | 2 posições *one-hot* |
| Idade | Valor dividido por 100 |
| Hábitos alimentares | 3 posições *one-hot* |
| Pressão académica | Valor dividido por 5 |
| Satisfação com os estudos | Valor dividido por 5 |
| Horas de estudo | Valor dividido por 12 |
| Stress financeiro | Valor dividido por 5 |

A função `translate(lista, normalizar)` produz um vetor de **10 elementos** (2 + 1 + 3 + 4) e um padrão com a estrutura:

```python
[entradas, classe_original, saida_codificada]
```

**Importante:** a tabela acima descreve a intenção da função. No CSV disponibilizado, `Dietary Habits` está na posição 5, mas a função procura-a na posição 2; outras posições também estão desalinhadas. Consequentemente, **o pré-processamento atual não representa corretamente todos estes atributos**.

## Treino e avaliação

O ciclo de treino executa, para cada estudante:

1. `forward()` — calcula as ativações das camadas escondida e de saída;
2. `error()` — calcula os sinais de erro da saída e da camada escondida;
3. `update()` — atualiza os pesos, utilizando a taxa `alpha`;
4. `iterate()` — reúne estas operações e imprime os valores de entrada, a saída pretendida e a saída produzida.

A função `test_students()` determina a **exatidão (*accuracy*)** do classificador:

```text
Accuracy (%) = (número de previsões corretas / número de exemplos) × 100
```

Durante o treino, `train_students()` mede a exatidão nos conjuntos de treino e teste no final de cada época e apresenta um gráfico da evolução de ambas. A experiência principal compara posteriormente a exatidão final entre configurações.

> O código denomina por vezes esta métrica «precisão», mas calcula **accuracy**, não *precision* no sentido técnico da classificação binária. Não calcula matriz de confusão, *recall* nem F1-score.

## Experiências executadas

Ao executar `nn_students_alunos.py`, o bloco principal percorre estas configurações:

| Parâmetro | Valores |
| --- | --- |
| Neurónios escondidos | `4`, `7`, `11` |
| Épocas | `5`, `50`, `100` |
| Treino/teste | `275/123`, `379/123` |
| Normalização | `True`, `False` |
| Neurónios de saída | `2` |
| Taxa de aprendizagem | `0.2` |

São realizadas **36 combinações** (`3 × 3 × 2 × 2`). Para cada combinação, o programa:

- constrói os conjuntos de treino e teste usando as primeiras linhas do CSV;
- cria e treina uma nova rede neuronal;
- apresenta as exatidões de treino e teste por época;
- mostra um gráfico com a evolução da exatidão;
- calcula a exatidão final no conjunto de teste;
- guarda os parâmetros e o resultado numa lista em memória (`resultados`).

No final, são apresentados gráficos comparativos para os dados normalizados e não normalizados. **Os gráficos são mostrados no ecrã (`plt.show()`), mas não são automaticamente guardados em ficheiros.**

## Instalação e execução

### Requisitos

- Python 3;
- biblioteca [`matplotlib`](https://matplotlib.org/).

As bibliotecas `random`, `math` e `csv` fazem parte da biblioteca padrão do Python.

Instalar a dependência externa:

```bash
python -m pip install matplotlib
```

Colocar o script e o CSV na **mesma pasta**, mantendo os nomes indicados na estrutura do repositório, e executar:

```bash
python nn_students_alunos.py
```

O programa escreve muitos detalhes na consola, porque `iterate()` imprime uma linha **por exemplo de treino e por época**, e abre várias janelas de gráficos. É necessário fechar as janelas para continuar a execução nos ambientes em que `plt.show()` é bloqueante.

## Principais funções

| Função | Responsabilidade |
| --- | --- |
| `make(nx, nz, ny)` | Criar a rede e inicializar os pesos |
| `sig(inp)` | Aplicar a função sigmoide |
| `forward(nn, input)` | Propagar as entradas até à saída |
| `error(nn, output)` | Calcular os sinais de erro para *backpropagation* |
| `update(nn)` | Atualizar pesos |
| `iterate(i, nn, input, output)` | Efetuar uma iteração de treino |
| `train_and()`, `train_or()`, `train_xor()` | Exemplos de treino de funções booleanas |
| `translate(lista, normalizar)` | Converter um registo num padrão de classificação |
| `build_sets(data_list, x, y)` | Baralhar uma lista e construir os conjuntos de treino e teste |
| `normaliza_valores(lista)` | Normalizar cinco atributos numéricos numa função auxiliar |
| `converte_categ_numerico(valor, categorias)` | Efetuar codificação *one-hot* genérica |
| `train_students(...)` | Treinar e representar a evolução da exatidão |
| `test_students(...)` | Calcular a exatidão num conjunto de dados |
| `retranslate(out)` | Converter as ativações em classe binária |
| `run_students(...)` | Função auxiliar de orquestração, não usada pelo bloco principal |

## Limitações conhecidas

1. **Índices incorretos no CSV:** a função `translate()` trata `lista[2]` como hábitos alimentares, embora corresponda a `Academic Pressure`; trata `lista[3]` como pressão académica, embora seja `Study Satisfaction`; e trata `lista[4]` como satisfação, embora seja `Sleep Duration`. Assim, a codificação dos hábitos alimentares resulta, neste dataset, em zeros, e a conversão numérica da duração do sono falha e utiliza `0.0`. A variável `Academic Pressure` não é usada corretamente.
2. **Dimensão de entrada inconsistente:** `translate()` produz **10 entradas**, mas o bloco principal chama `train_students(18, ...)`. Como `forward()` utiliza `zip()`, o cálculo efetivo usa apenas os pesos correspondentes às entradas disponíveis, deixando pesos de entrada excedentes sem utilização. A implementação deveria harmonizar estas dimensões.
3. **Variáveis ignoradas:** `Have you ever had suicidal thoughts ?` e `Family History of Mental Illness` não são integradas no vetor de entrada. A coluna `Sleep Duration` também não é codificada como uma característica válida na implementação atual.
4. **Argumentos ignorados em `run_students()`:** apesar de receber parâmetros configuráveis, esta função chama `build_sets(data, 384, 128)` e `train_students(18, 5, 2, ..., 5)` com valores fixos. Além disso, `build_sets()` espera uma lista, não um caminho de ficheiro, e `translate()` exige o parâmetro `normalizar`, que essa função não fornece. Por isso, **`run_students()` não está funcional tal como se encontra**, embora não seja chamada na execução principal.
5. **Ausência de reprodutibilidade:** não é definida uma semente aleatória (*random seed*); por isso, pesos iniciais e resultados podem variar entre execuções.
6. **Conjunto de teste reutilizado durante o treino:** o desempenho é acompanhado no mesmo conjunto de teste em todas as épocas. Para comparar modelos e selecionar configurações com rigor, deve existir um conjunto de validação separado e um teste final independente.
7. **Partições sequenciais na experiência principal:** o bloco `__main__` não baralha os registos antes de os dividir; utiliza fatias consecutivas da ordem original do CSV. A função `build_sets()` inclui baralhamento, mas não é utilizada nessa experiência.
8. **Métricas limitadas:** apenas a exatidão é calculada. Em classificação relacionada com saúde mental, seria especialmente importante avaliar também falsos negativos, falsos positivos, *precision*, *recall* e F1-score.
9. **Ausência de validação clínica:** o dataset e os resultados do modelo não justificam a sua utilização para diagnosticar depressão ou tomar decisões sobre estudantes.

## Melhorias futuras

- Corrigir os índices das colunas em `translate()` e fazer a codificação adequada das características selecionadas.
- Ajustar `input_size` ao número efetivo de características geradas.
- Corrigir `run_students()` para respeitar os argumentos recebidos.
- Adicionar `random.seed(...)` e uma divisão aleatória, estratificada e reproduzível dos dados.
- Separar treino, validação e teste final.
- Acrescentar matriz de confusão, *precision*, *recall* e F1-score.
- Reduzir os logs e permitir guardar automaticamente os gráficos e resultados.
- Avaliar eventuais enviesamentos do dataset e os limites de generalização do modelo.

## Tecnologias

- **Python** — implementação da lógica da rede neuronal e manipulação do CSV;
- **Matplotlib** — visualização da evolução e comparação do desempenho;
- **Biblioteca padrão do Python** (`csv`, `random`, `math`) — leitura dos dados, inicialização aleatória e operações matemáticas.

---

**Âmbito:** projeto académico na e experimental de redes neuronais e classificação binária. A documentação descreve o código disponibilizado, incluindo os problemas identificados; não pressupõe que tenham sido corrigidos nem apresenta métricas de desempenho que não tenham sido medidas.
