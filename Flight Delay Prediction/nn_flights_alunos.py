"""
Luís Laia, nº 20181127
Rodrigo Lopes, nº 20230321
"""


import random
import math
import csv
import matplotlib.pyplot as plt


#------------------CÓDIGO GENÉRICO PARA CRIAR, TREINAR E USAR UMA REDE COM UMA CAMADA ESCONDIDA------------
def make(nx, nz, ny):
    """Funcao que cria, inicializa e devolve uma rede neuronal, incluindo
    a criacao das diversos listas, bem como a inicializacao das listas de pesos. 
    Note-se que sao incluidas duas unidades extra, uma de entrada e outra escondida, 
    mais os respectivos pesos, para lidar com os tresholds; note-se tambem que, 
    tal como foi discutido na teorica, as saidas destas unidades estao sempre a -1.
    por exemplo, a chamada make(3, 5, 2) cria e devolve uma rede 3x5x2"""
    #a rede neuronal é num dicionario com as seguintes chaves:
    # nx     numero de entradas
    # nz     numero de unidades escondidas
    # ny     numero de saidas
    # x      lista de armazenamento dos valores de entrada
    # z      array de armazenamento dos valores de activacao das unidades escondidas
    # y      array de armazenamento dos valores de activacao das saidas
    # wzx    array de pesos entre a camada de entrada e a camada escondida
    # wyz    array de pesos entre a camada escondida e a camada de saida
    # dz     array de erros das unidades escondidas
    # dy     array de erros das unidades de saida    
    
    nn = {'nx':nx, 'nz':nz, 'ny':ny, 'x':[], 'z':[], 'y':[], 'wzx':[], 'wyz':[], 'dz':[], 'dy':[]}
    
    nn['wzx'] = [[random.uniform(-0.5,0.5) for _ in range(nn['nx'] + 1)] for _ in range(nn['nz'])]
    nn['wyz'] = [[random.uniform(-0.5,0.5) for _ in range(nn['nz'] + 1)] for _ in range(nn['ny'])]
    
    return nn

def sig(inp):
    """Funcao de activacao (sigmoide)"""
    return 1.0/(1.0 + math.exp(-inp))


def forward(nn, input):
    """Função que recebe uma rede nn e um padrao de entrada input (uma lista) 
    e faz a propagacao da informacao para a frente ate as saidas"""
    #copia a informacao do vector de entrada input para a listavector de inputs da rede nn  
    nn['x']=input.copy()
    #adiciona a entrada a -1 que vai permitir a aprendizagem dos limiares
    nn['x'].append(-1)
    #calcula a activacao da unidades escondidas
    for i in range (nn['nz']):
        nn['z']=[sig(sum([x*w for x, w in zip(nn['x'], nn['wzx'][i])])) for i in range(nn['nz'])]
        #adiciona a entrada a -1 que vai permitir a aprendizagem dos limiares
        nn['z'].append(-1)
        #calcula a activacao da unidades de saida
        nn['y']=[sig(sum([z*w for z, w in zip(nn['z'], nn['wyz'][i])])) for i in range(nn['ny'])]
 
   
def error(nn, output):
    """Funcao que recebe uma rede nn com as activacoes calculadas
       e a lista output de saidas pretendidas e calcula os erros
       na camada escondida e na camada de saida"""
    nn['dy']=[y*(1-y)*(o-y) for y,o in zip(nn['y'], output)]
    zerror=[sum([nn['wyz'][i][j]*nn['dy'][i] for i in range(nn['ny'])]) for j in range(nn['nz'])]
    nn['dz']=[z*(1-z)*e for z, e in zip(nn['z'], zerror)]
 
 
def update(nn):
    """funcao que recebe uma rede com as activacoes e erros calculados e
    actualiza as listas de pesos"""
    
    nn['wzx'] = [ [w+x*nn['dz'][i]*alpha for w, x in zip(nn['wzx'][i], nn['x'])] for i in range(nn['nz'])]
    nn['wyz'] = [ [w+z*nn['dy'][i]*alpha for w, z in zip(nn['wyz'][i], nn['z'])] for i in range(nn['ny'])]
    

def iterate(i, nn, input, output):
    """Funcao que realiza uma iteracao de treino para um dado padrao de entrada input
    com saida desejada output"""
    forward(nn, input)
    error(nn, output)
    update(nn)
    print('%03i: %s -----> %s : %s' %(i, input, output, nn['y']))
    

#-----------------------CÓDIGO QUE PERMITE CRIAR E TREINAR REDES PARA APRENDER AS FUNÇÕES BOOLENAS--------------------
"""Funcao que cria uma rede 2x2x1 e treina a função lógica AND
A função recebe como entrada o número de épocas com que se pretende treinar a rede"""
def train_and(epocas):
    net = make(2, 2, 1)
    for i in range(epocas):
        iterate(i, net, [0, 0], [0])
        iterate(i, net, [0, 1], [0])
        iterate(i, net, [1, 0], [0])
        iterate(i, net, [1, 1], [1])
    return net
    
"""Funcao que cria uma rede 2x2x1 e treina um OR
A função recebe como entrada o número de épocas com que se pretende treinar a rede"""
def train_or(epocas):
    net = make(2, 2, 1)
    for i in range(epocas):
        iterate(i, net, [0, 0], [0])
        iterate(i, net, [0, 1], [1])
        iterate(i, net, [1, 0], [1])
        iterate(i, net, [1, 1], [1]) 
    return net

"""Funcao que cria uma rede 2x2x1 e treina um XOR
A função recebe como entrada o número de épocas com que se pretende treinar a rede"""
def train_xor(epocas):
    net = make(2, 2, 1)
    for i in range(epocas):
        iterate(i, net, [0, 0], [0])
        iterate(i, net, [0, 1], [1])
        iterate(i, net, [1, 0], [1])
        iterate(i, net, [1, 1], [0]) 
    return net


#-------------------------CÓDIGO QUE IRÁ PERMITIR CRIAR UMA REDE PARA APRENDER A CLASSIFICAR VOOS DE AVIÃO---------    

# Dicionário com listas
META = {
    "carriers": [],
    "origins": [],
    "dests": [],
    "dist_min": None,
    "dist_max": None
}

"""
Esta função converte um valor de tempo no formato HHMM para o número
total de minutos desde a meia-noite, facilitando o uso de dados
temporais em cálculos e análises.
"""
def _hhmm_para_minutos(hhmm: int) -> int:
    hh = hhmm // 100
    mm = hhmm % 100
    return hh * 60 + mm


"""
Esta função lê um arquivo CSV, extrai informações sobre transportadoras,
origens, destinos e distâncias, e atualiza o dicionário META com essas
informações organizadas e prontas para uso no treinamento da rede.
"""
def _preparar_meta(nome_f: str) -> None:
    carriers, origins, dests = set(), set(), set()
    dist_min, dist_max = None, None

    with open(nome_f, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        _ = next(reader, None)

        for row in reader:
            if not row or len(row) < 8:
                continue

            carrier = row[3].strip()
            origin = row[4].strip()
            dest = row[5].strip()

            distance = int(float(row[6]))

            carriers.add(carrier)
            origins.add(origin)
            dests.add(dest)

            dist_min = distance if dist_min is None else min(dist_min, distance)
            dist_max = distance if dist_max is None else max(dist_max, distance)

    META["carriers"] = sorted(list(carriers))
    META["origins"]  = sorted(list(origins))
    META["dests"]    = sorted(list(dests))
    META["dist_min"] = dist_min
    META["dist_max"] = dist_max


"""
Esta função calcula e retorna o número total de entradas que a rede neuronal
deve receber, levando em consideração os atributos como transportadoras,
origens e destinos.
"""
def get_input_size() -> int:
    return 4 + len(META["carriers"]) + len(META["origins"]) + len(META["dests"])


"""Funcao principal do nosso programa para prever se um voo chegará com ou sem atraso:
cria os conjuntos de treino e teste, chama a funcao que cria e treina a rede e, por fim, 
a funcao que a testa. A funcao recebe como argumento o ficheiro correspondente ao dataset 
que deve ser usado, os tamanhos das camadas de entrada, escondida e saída,
o numero de epocas que deve ser considerado no treino e os tamanhos dos conjuntos de treino e 
teste"""
def run_flights(file, input_size, hidden_size, output_size, epochs, training_set_size, test_set_size):
    training_set, test_set = build_sets(file, training_set_size, test_set_size)

    true_input = get_input_size()
    if input_size != true_input:
        input_size = true_input

    
    net = train_flights(
        input_size,
        hidden_size,
        output_size,
        training_set,
        test_set,
        epochs
    )
    return net


"""Funcao que cria os conjuntos de treino e de teste a partir dos dados
armazenados em f ('DataSet1.csv'). A funcao le cada linha, 
tranforma-a numa lista de valores e chama a funcao translate para a colocar no 
formato adequado para o padrao de treino. Estes padroes são colocados numa lista.
A função recebe como argumentos o nº de exemplos que devem ser considerados no conjunto de 
treino --->x e o nº de exemplos que devem ser considerados no conjunto de teste ------> y
Finalmente, devolve duas listas, uma com x padroes (conjunto de treino)
e a segunda com y padrões (conjunto de teste). Atenção que x+y não pode ultrapassar o nº 
de estudantes disponível no dataset"""       
def build_sets(nome_f, x, y):
    _preparar_meta(nome_f)

    padroes = []
    with open(nome_f, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        _ = next(reader, None)  # ignora o header

        for row in reader:
            if not row or len(row) < 8:
                continue

            linha = [
                int(row[0]),            # Month
                int(row[1]),            # DayOfWeek
                int(row[2]),            # CRSDepTime (HHMM)
                row[3].strip(),         # UniqueCarrier
                row[4].strip(),         # Origin
                row[5].strip(),         # Dest
                int(float(row[6])),     # Distance
                int(row[7])             # Delayed
            ]
            padroes.append(translate(linha))

    if x + y > len(padroes):
        raise ValueError(f"x+y ({x+y}) > nº exemplos no dataset ({len(padroes)})")

    random.shuffle(padroes)
    return padroes[:x], padroes[x:x+y]


"""A função translate recebe cada lista de valores que caracterizam um voo
e transforma-a num padrão de treino. Cada padrão é uma lista com o seguinte formato 
[padrao_de_entrada, classe_do_voo, padrao_de_saida]
O enunciado do trabalho explica de que forma deve ser obtido o padrão de entrada
"""
def translate(lista):
    month = int(lista[0])
    day = int(lista[1])
    crs_hhmm = int(lista[2])
    carrier = str(lista[3]).strip()
    origin = str(lista[4]).strip()
    dest = str(lista[5]).strip()
    distance = int(lista[6])
    delayed = int(lista[7])

    # HHMM -> minutos
    dep_minutes = _hhmm_para_minutos(crs_hhmm)

    # normalização numéricos
    norm_vals = normaliza_valores([month, day, dep_minutes, distance])
    numericos = [norm_vals[0], norm_vals[1], norm_vals[2], norm_vals[3]]

    # one-hot categóricos
    oh_carrier = converte_categ_numerico((carrier, META["carriers"]))
    oh_origin  = converte_categ_numerico((origin,  META["origins"]))
    oh_dest    = converte_categ_numerico((dest,    META["dests"]))

    x = numericos + oh_carrier + oh_origin + oh_dest

    # saída (classe)
    y = [1, 0] if delayed == 0 else [0, 1]

    return [x, delayed, y]


#Função que converte valores categóricos para a codificação onehot                
def converte_categ_numerico(instancia):
    valor, universo = instancia
    onehot = [0] * len(universo)
    if valor in universo:
        onehot[universo.index(valor)] = 1
    return onehot



"""Função que normaliza os valores necessários"""   
def normaliza_valores(lista):
    month, day, dep_minutes, distance = lista

    if not NORMALIZE:
        return [float(month), float(day), float(dep_minutes), float(distance)]

    # Month 1..12
    month_norm = (month - 1) / (12 - 1)

    # DayOfWeek 1..7
    day_norm = (day - 1) / (7 - 1)

    # CRSDepTime(min) 0..1439
    dep_norm = dep_minutes / 1439

    # Distancia min/max do dataset
    dmin, dmax = META["dist_min"], META["dist_max"]
    if dmin is None or dmax is None or dmax == dmin:
        dist_norm = 0.0
    else:
        dist_norm = (distance - dmin) / (dmax - dmin)

    return [month_norm, day_norm, dep_norm, dist_norm]



"""Cria a rede e chama a funçao iterate para a treinar. A função recebe como argumento 
o conjunto de treino, os tamanhos das camadas de entrada, escondida e saída e o número 
de épocas que irão ser usadas para fazer o treino"""
def train_flights(input_size, hidden_size, output_size, training_set, test_set, epochs):
    """
    Cria a rede, treina e gera gráfico com metadados da experiência.
    """
    net = make(input_size, hidden_size, output_size)
    
    # Histórico das métricas
    history_train_acc = []
    history_test_acc = []
    history_precision = []
    history_recall = []
    history_f1 = []
    epochs_list = []
    
    print(f"A iniciar treino por {epochs} épocas (Alpha={alpha})...")
    print_interval = max(1, epochs // 10)

    for i in range(epochs):
        random.shuffle(training_set)
        
        # --- TREINO ---
        acertos_treino = 0
        for padrao in training_set:
            inputs = padrao[0]
            target_vector = padrao[2]
            target_class = padrao[1]
            
            forward(net, inputs)
            
            # Accuracy Treino
            out = net['y']
            prev = 0 if out[0] > out[1] else 1
            if prev == target_class:
                acertos_treino += 1
            
            error(net, target_vector)
            update(net)
            
        acc_treino = acertos_treino / len(training_set)
        
        # --- TESTE ---
        metrics = test_flights(net, test_set, printing=True)
        acc_teste = metrics["exatidão"]
        precision = metrics["precisão"]
        recall = metrics["cobertura"]
        f1 = metrics["F1-score"]
        
        # Armazenar as métricas
        history_train_acc.append(acc_treino)
        history_test_acc.append(acc_teste)
        history_precision.append(precision)
        history_recall.append(recall)
        history_f1.append(f1)
        epochs_list.append(i)
        
        if (i+1) % print_interval == 0:
            print(f"Época {i+1}/{epochs} | Treino: {acc_treino:.2%} | Teste: {acc_teste:.2%}")

    # --- GRÁFICOS COM DADOS DA EXPERIÊNCIA ---
    plt.figure(figsize=(14, 7))
    plt.subplots_adjust(right=0.75) 
    
    plt.plot(epochs_list, history_train_acc, label='Treino', linewidth=2)
    plt.plot(epochs_list, history_test_acc, label='Teste', linewidth=2)    
    plt.plot(epochs_list, history_precision, label='Precisão', linestyle='-', linewidth=2)
    plt.plot(epochs_list, history_recall, label='Cobertura', linestyle='-', linewidth=2)
    plt.plot(epochs_list, history_f1, label='F1-Score', linestyle='-', linewidth=2)
    
    
    plt.xlabel('Épocas')
    plt.ylabel('Exatidão')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    
    # Texto com os Parâmetros da Experiência
    texto_params = (
        f"DADOS DA EXPERIÊNCIA\n"
        f"----------------------------------------\n"
        f"Neurónios Escondidos: {hidden_size}\n"
        f"Épocas: {epochs}\n"
        f"Alpha: {alpha}\n"
        f"Treino: {len(training_set)} \n"
        f"Teste: {len(test_set)} \n"
        f"Normalização: ativa (Min-Max)\n"
        f"Atributos: Todos "
    )
    
   
    plt.figtext(0.78, 0.5, texto_params, fontsize=10, 
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'),
                verticalalignment='center')
    
    nome_grafico = "teste_Gráfico.png"
    plt.savefig(nome_grafico)
    print(f"Gráfico guardado como '{nome_grafico}'")
    
    return net


"""Funcao que avalia a precisao da rede treinada, utilizando o conjunto de teste ou treino.
Para cada padrao do conjunto chama a funcao forward e determina a classe do voo
que corresponde ao maior valor da lista de saida. A classe determinada pela rede
deve ser comparada com a classe real,sendo contabilizado o número de respostas corretas. 
A função calcula a percentagem de respostas corretas, 
o nº de VP,FP,VN, FN, precisão, cobertura e f1-score""" 
def test_flights(net, test_set, printing=True):
    """
    Avalia a precisão da rede treinada utilizando o conjunto de teste.
    Calcula VP, FP, VN, FN, precisão, cobertura e f1-score.
    """
    vp, vn, fp, fn = 0, 0, 0, 0

    for idx, padrao in enumerate(test_set):
        # padrao vem do translate: [inputs, classe_real_int, target_vector]
        inputs = padrao[0]
        classe_real = padrao[1]  # 0 ou 1
        
        # Propagar inputs pela rede
        forward(net, inputs)
        outputs = net['y']
        
        # Determinar previsão (índice do maior valor)
        # 0 = Sem Atraso, 1 = Com Atraso
        classe_prevista = 0 if outputs[0] > outputs[1] else 1
        
        # Matriz de Confusão
        if classe_prevista == 1 and classe_real == 1:
            vp += 1
        elif classe_prevista == 0 and classe_real == 0:
            vn += 1
        elif classe_prevista == 1 and classe_real == 0:
            fp += 1
        elif classe_prevista == 0 and classe_real == 1:
            fn += 1
        
        # Impressão dos resultados dos padrões
        if printing:
            real_status = "atrasado" if classe_real == 1 else "não atrasado"
            predicted_status = "atrasado" if classe_prevista == 1 else "não atrasado"
            print(f"A rede prevê que o voo nº {idx+1} {predicted_status} e na realidade ele está {real_status}.")

    # Cálculos das métricas (com proteção contra divisão por zero)
    total = vp + vn + fp + fn
    accuracy = (vp + vn) / total if total > 0 else 0
    precision = vp / (vp + fp) if (vp + fp) > 0 else 0
    recall = vp / (vp + fn) if (vp + fn) > 0 else 0 # Cobertura
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # Impressão do desempenho final da rede
    if printing:
        print("-" * 30)
        print(f"RESULTADOS DO TESTE (Total: {total} voos)")
        print("-" * 30)
        print(f"Matriz de Confusão:")
        print(f"VP (Verdadeiros Positivos): {vp}")
        print(f"VN (Verdadeiros Negativos): {vn}")
        print(f"FP (Falsos Positivos): {fp}")
        print(f"FN (Falsos Negativos): {fn}")
        print("-" * 30)
        print(f"Exatidão:  {accuracy:.2%}")
        print(f"Precisão:  {precision:.2%}")
        print(f"Cobertura: {recall:.2%}")
        print(f"F1-Score:  {f1:.2%}")
        print("-" * 30)
        print(f"Success rate: {accuracy * 100:.2f}")

    # Criar a imagem da tabela com os resultados
    nome_imagem = 'teste_Tabela.png'
    tabela_resultados(
        {
            "exatidão": accuracy,
            "precisão": precision,
            "cobertura": recall,
            "F1-score": f1,
            "matriz": [vp, fp, vn, fn]
        },
        nome_imagem
    )
    
    # Retorna um dicionário com as métricas calculadas
    return {
        "exatidão": accuracy,
        "matriz": [vp, fp, vn, fn],
        "precisão": precision,
        "cobertura": recall,
        "F1-score": f1
    }

def tabela_resultados(metrics, nome_imagem):
    """
    Função para guardar os resultados de métricas em uma tabela em imagem
    'metrics' é um dicionário com as chaves e valores das métricas
    """
    # dados da tabela
    tabela = [
        ["Métrica", "Valor"],
        ["Exatidão", f"{metrics['exatidão']:.2%}"],
        ["Precisão", f"{metrics['precisão']:.2%}"],
        ["Cobertura", f"{metrics['cobertura']:.2%}"],
        ["F1-Score", f"{metrics['F1-score']:.2%}"],
        ["VP", metrics['matriz'][0]],
        ["VN", metrics['matriz'][1]],
        ["FP", metrics['matriz'][2]],
        ["FN", metrics['matriz'][3]],
        ["Success rate", f"{metrics['exatidão']*100:.2f}%"],
    ]
    
    # Cria a imagem com a tabela
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis('tight')
    ax.axis('off')
    
    # Criar a tabela
    table = ax.table(cellText=tabela, loc='center', cellLoc='center', colWidths=[0.3, 0.3])
    
    # Melhorar a aparência da tabela
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.5, 1.5)

    # guarda a imagem
    plt.savefig(nome_imagem, bbox_inches='tight', pad_inches=0.1)
    print(f"Tabela guardada como '{nome_imagem}'")
    plt.close()
  
"""Recebe o padrao de saida da rede e devolve a situação de atraso do voo.
A situação de atraso corresponde ao indice da saida com maior valor."""  
def retranslate(out):
    """
    Recebe o vetor de saída da rede (ex: [0.92, 0.08]) e devolve a classe legível.
    Baseado na regra:
    - Índice 0 maior -> Classe 0 ("não atrasado")
    - Índice 1 maior -> Classe 1 ("atrasado")
    """
    # Se o valor do primeiro neurónio for maior que o do segundo
    if out[0] > out[1]:
        return "não atrasado"  # Classe 0
    else:
        return "atrasado"      # Classe 1


alpha = 0.1 #velocidade de aprendizagem
NORMALIZE = True


if __name__ == "__main__":
    #Vamos treinar durante 1000 épocas uma rede para aprender a função logica AND
    #Faz testes para números de épocas diferentes e para as restantes funções lógicas já implementadas
    rede_AND = train_and(1000)
    #Agora vamos ver se ela aprendeu bem
    tabela_verdade = {(0,0): 0, (0,1): 0, (1,0): 0, (1,1): 1}
    for linha in tabela_verdade:
        forward(rede_AND, list(linha))
        print('A rede determinou %s para a entrada %d AND %d quando devia ser %d'
              %(rede_AND['y'], linha[0], linha[1], tabela_verdade[linha]))
        

if __name__ == "__main__":
    input_size, hidden_size, output_size = 18, 8, 2
    epochs = 50
    training_set_size = 800
    test_set_size = 200
    file = 'flights_simulated_balanced_1000.csv'
    run_flights(file, input_size, hidden_size, output_size, epochs, training_set_size, test_set_size)