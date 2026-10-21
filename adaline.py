"""
ADALINE (ADAptive LInear NEuron) treinado pelo algoritmo LMS (Widrow-Hoff).

Implementação baseada no capítulo 10 de "Neural Network Design"
(Hagan, Demuth, Beale e De Jesús), usando a mesma notação do livro:

    p     vetor de entrada (R x 1)
    t     saída alvo / target (S x 1)
    n     entrada líquida, n = Wp + b (S x 1)
    a     saída da rede, a = purelin(n) (S x 1)
    e     erro, e = t - a (S x 1)
    W     matriz de pesos (S x R); a linha i é o vetor iw transposto
    b     vetor de bias (S x 1)
    alpha taxa de aprendizagem (α no livro)
    k     iteração do algoritmo LMS
    x     vetor de parâmetros de um neurônio, x = [1w; b]       (Eq. 10.6)
    z     vetor de entrada aumentado, z = [p; 1]                 (Eq. 10.7)
    c     c = E[t^2]                                             (Eq. 10.13)
    h     correlação cruzada entrada/alvo, h = E[t z]            (Eq. 10.13)
    R     matriz de correlação da entrada, R = E[z z^T]          (Eq. 10.13)

Como no livro, os conjuntos de dados são organizados por colunas: cada coluna
é um padrão q = 1, ..., Q.

    P     matriz de entradas (R x Q), colunas p_q
    T     matriz de alvos (S x Q), colunas t_q
"""
import numpy as np

from metricas import mse


def purelin(n):
    """Função de transferência linear: a = purelin(n) = n."""
    return n


class ADALINE:
    """Rede ADALINE com S neurônios lineares e R entradas (Figura 10.1)."""

    def __init__(self, R, S=1, alpha=0.01, inicializacao="zeros", semente=None):
        if alpha <= 0:
            raise ValueError("A taxa de aprendizagem alpha deve ser positiva.")

        self.R = R
        self.S = S
        self.alpha = alpha
        self.k = 0
        self.gerador = np.random.default_rng(semente)

        if inicializacao == "zeros":
            # O livro inicia os exemplos com todos os pesos iguais a zero.
            self.W = np.zeros((S, R))
            self.b = np.zeros((S, 1))
        elif inicializacao == "aleatoria":
            self.W = self.gerador.uniform(-0.5, 0.5, size=(S, R))
            self.b = self.gerador.uniform(-0.5, 0.5, size=(S, 1))
        else:
            raise ValueError("inicializacao deve ser 'zeros' ou 'aleatoria'.")

    def simular(self, P):
        """Saída da rede para cada coluna de P: a = purelin(Wp + b) (Eq. 10.1)."""
        n = self.W @ P + self.b
        return purelin(n)

    def passo_lms(self, p, t):
        """
        Uma iteração k do algoritmo LMS para o par {p(k), t(k)}:

            a(k)   = purelin(W(k) p(k) + b(k))
            e(k)   = t(k) - a(k)
            W(k+1) = W(k) + 2 alpha e(k) p(k)^T                     (Eq. 10.33)
            b(k+1) = b(k) + 2 alpha e(k)                            (Eq. 10.34)

        A atualização usa o gradiente estocástico
        grad_hat F(x) = grad e^2(k) = -2 e(k) z(k) (Eq. 10.26).
        """
        a = purelin(self.W @ p + self.b)
        e = t - a
        self.W = self.W + 2 * self.alpha * e @ p.T
        self.b = self.b + 2 * self.alpha * e
        self.k += 1
        return e

    def treinar(self, P, T, epocas, P_teste=None, T_teste=None, embaralhar=True):
        """
        Treina a rede com o algoritmo LMS (aprendizagem incremental / on-line).

        Em cada época os Q pares {p_q, t_q} de treinamento são apresentados uma
        vez, um por iteração k. O livro observa que a ordem de apresentação
        pode ser aleatória, por isso, por padrão, a ordem é embaralhada a cada
        época.

        Ao final de cada época registra-se o erro quadrático médio
        F(x) = E[e^2] (Eq. 10.10), estimado pela média sobre o conjunto de
        treinamento e, se fornecido, sobre o conjunto de teste.
        """
        Q = P.shape[1]
        historico = {"mse_treino": [], "mse_teste": []}

        for _ in range(epocas):
            ordem = self.gerador.permutation(Q) if embaralhar else np.arange(Q)
            for q in ordem:
                p = P[:, [q]]
                t = T[:, [q]]
                self.passo_lms(p, t)

            historico["mse_treino"].append(mse(T, self.simular(P)))
            if P_teste is not None and T_teste is not None:
                historico["mse_teste"].append(mse(T_teste, self.simular(P_teste)))

        return historico

    def vetor_x(self, i=0):
        """Parâmetros do neurônio i agrupados em x = [iw; b_i] (Eq. 10.6)."""
        return np.concatenate([self.W[i], self.b[i]])


def vetor_z(P):
    """Inclui a entrada de bias "1" em cada padrão: z = [p; 1] (Eq. 10.7)."""
    Q = P.shape[1]
    return np.vstack([P, np.ones((1, Q))])


def estatisticas_correlacao(P, t):
    """
    Estima c, h e R (Eq. 10.13) para um único neurônio.

    A esperança E[.] é substituída pela média sobre os Q padrões, como
    nos exemplos do livro em que cada padrão ocorre com a mesma probabilidade:

        c = E[t^2],   h = E[t z],   R = E[z z^T]
    """
    t = np.asarray(t, dtype=float).ravel()
    Z = vetor_z(P)
    Q = Z.shape[1]
    c = np.mean(t ** 2)
    h = Z @ t / Q
    R = Z @ Z.T / Q
    return c, h, R


def taxa_aprendizagem_maxima(P):
    """
    Limite de estabilidade do LMS: 0 < alpha < 1 / lambda_max (Eq. 10.44),
    em que lambda_max é o maior autovalor da matriz de correlação R.

    Retorna (1 / lambda_max, autovalores de R).
    """
    Z = vetor_z(P)
    R = Z @ Z.T / Z.shape[1]
    autovalores = np.linalg.eigvalsh(R)  # R é simétrica
    return 1.0 / autovalores.max(), autovalores


def indice_desempenho(x, c, h, R):
    """Erro quadrático médio como função quadrática: F(x) = c - 2 x^T h + x^T R x (Eq. 10.12)."""
    return float(c - 2 * x @ h + x @ R @ x)


def solucao_minimo_mse(P, t):
    """
    Ponto de mínimo do erro quadrático médio, x* = R^-1 h (Eq. 10.18).

    Não é usado no treinamento: serve de referência para verificar que o
    LMS converge para a solução de mínimo MSE (Eq. 10.46).
    Retorna (x*, F(x*)).
    """
    c, h, R = estatisticas_correlacao(P, t)
    x_otimo = np.linalg.solve(R, h)
    return x_otimo, indice_desempenho(x_otimo, c, h, R)
