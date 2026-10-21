"""
Leitura e pré-processamento da base Auto MPG (UCI, dataset 9).

Colunas do arquivo auto-mpg.data:
    1. mpg           contínua (saída alvo t)
    2. cylinders     discreta multivalorada
    3. displacement  contínua
    4. horsepower    contínua (6 valores ausentes, marcados com "?")
    5. weight        contínua
    6. acceleration  contínua
    7. model_year    discreta multivalorada
    8. origin        discreta multivalorada (1 = EUA, 2 = Europa, 3 = Japão)
    9. car_name      texto (único para cada instância, não usado como entrada)

As matrizes devolvidas seguem a convenção do livro "Neural Network Design":
cada coluna é um padrão, P tem dimensão R x Q e T tem dimensão 1 x Q.
"""
import numpy as np

COLUNAS = [
    "mpg",
    "cylinders",
    "displacement",
    "horsepower",
    "weight",
    "acceleration",
    "model_year",
    "origin",
    "car_name",
]

ENTRADAS_NUMERICAS = [
    "cylinders",
    "displacement",
    "horsepower",
    "weight",
    "acceleration",
    "model_year",
]

# A origem é categórica, então vira variáveis binárias. Os EUA (origin = 1)
# ficam como categoria de referência: com as três categorias, a soma das
# variáveis binárias seria sempre 1, igual à entrada de bias, e a matriz de
# correlação R seria singular (autovalor nulo, sem mínimo único - Eq. 10.18).
ENTRADAS_ORIGEM = {"origin_europe": 2, "origin_japan": 3}

NOMES_ENTRADAS = ENTRADAS_NUMERICAS + list(ENTRADAS_ORIGEM)


def carregar_auto_mpg(caminho):
    """
    Lê o arquivo auto-mpg.data.

    Cada linha tem 8 campos numéricos separados por espaços, uma tabulação e o
    nome do carro entre aspas. Valores ausentes ("?") viram NaN.
    Retorna (matriz numérica Q x 8, lista com os nomes dos carros).
    """
    valores = []
    nomes = []
    with open(caminho, encoding="utf-8") as arquivo:
        for linha in arquivo:
            if not linha.strip():
                continue
            campos_numericos, nome = linha.rstrip("\n").split("\t", 1)
            valores.append(
                [np.nan if campo == "?" else float(campo) for campo in campos_numericos.split()]
            )
            nomes.append(nome.strip().strip('"'))
    return np.array(valores), nomes


def separar_treino_teste(Q, fracao_teste=0.2, semente=42):
    """Sorteia os índices dos conjuntos de treinamento e de teste."""
    gerador = np.random.default_rng(semente)
    indices = gerador.permutation(Q)
    Q_teste = int(round(fracao_teste * Q))
    return indices[Q_teste:], indices[:Q_teste]


def preparar_dados(caminho, fracao_teste=0.2, semente=42):
    """
    Monta os conjuntos de treinamento e teste.

    Os parâmetros do pré-processamento (mediana para valores ausentes, média e
    desvio padrão da padronização) são calculados somente com o conjunto de
    treinamento e depois aplicados ao de teste, para que nenhuma informação
    do teste seja usada no aprendizado.

    A padronização (média 0 e desvio 1) deixa as entradas na mesma escala.
    Isso reduz a dispersão dos autovalores de R e permite usar uma taxa de
    aprendizagem única que seja estável (Eq. 10.44) e eficiente para todos os
    pesos. A saída alvo t (mpg) é mantida na unidade original.

    Retorna um dicionário com P_treino, T_treino, P_teste, T_teste (notação do
    livro), os nomes das entradas e os nomes dos carros de cada conjunto.
    """
    dados, nomes_carros = carregar_auto_mpg(caminho)
    indice = {nome: j for j, nome in enumerate(COLUNAS[:-1])}

    t = dados[:, indice["mpg"]]
    entradas_numericas = dados[:, [indice[nome] for nome in ENTRADAS_NUMERICAS]]
    origem = dados[:, indice["origin"]]
    entradas_origem = np.column_stack(
        [(origem == codigo).astype(float) for codigo in ENTRADAS_ORIGEM.values()]
    )
    X = np.hstack([entradas_numericas, entradas_origem])  # Q x R

    treino, teste = separar_treino_teste(len(t), fracao_teste, semente)
    X_treino, X_teste = X[treino], X[teste]

    # Valores ausentes (horsepower) recebem a mediana do treinamento.
    mediana = np.nanmedian(X_treino, axis=0)
    X_treino = np.where(np.isnan(X_treino), mediana, X_treino)
    X_teste = np.where(np.isnan(X_teste), mediana, X_teste)

    # Padronização z-score com estatísticas do treinamento.
    media = X_treino.mean(axis=0)
    desvio = X_treino.std(axis=0)
    X_treino = (X_treino - media) / desvio
    X_teste = (X_teste - media) / desvio

    return {
        "P_treino": X_treino.T,           # R x Q_treino
        "T_treino": t[treino][None, :],   # 1 x Q_treino
        "P_teste": X_teste.T,             # R x Q_teste
        "T_teste": t[teste][None, :],     # 1 x Q_teste
        "nomes_entradas": NOMES_ENTRADAS,
        "carros_treino": [nomes_carros[i] for i in treino],
        "carros_teste": [nomes_carros[i] for i in teste],
        "media": media,
        "desvio": desvio,
    }
