"""
Métricas de avaliação de regressão.

Em todas as funções, t é a saída alvo (valor real) e a é a saída da rede
(valor previsto), seguindo a notação do capítulo 10 de "Neural Network Design".
O erro (resíduo) de cada padrão é e = t - a.
"""
import math

import numpy as np


def mse(t, a):
    """Mean Square Error: MSE = (1/Q) * soma(e_q^2)."""
    e = np.asarray(t, dtype=float) - np.asarray(a, dtype=float)
    return float(np.mean(e ** 2))


def rmse(t, a):
    """Root Mean Squared Error: RMSE = sqrt(MSE), na mesma unidade de t."""
    return math.sqrt(mse(t, a))


def mae(t, a):
    """Mean Absolute Error: MAE = (1/Q) * soma(|e_q|)."""
    e = np.asarray(t, dtype=float) - np.asarray(a, dtype=float)
    return float(np.mean(np.abs(e)))


def r2(t, a):
    """
    Coeficiente de determinação:

        R^2 = 1 - SS_res / SS_tot
        SS_res = soma((t_q - a_q)^2)
        SS_tot = soma((t_q - média(t))^2)

    R^2 = 1 indica previsões perfeitas; R^2 = 0 equivale a prever sempre a
    média de t; R^2 < 0 indica um modelo pior que essa previsão constante.
    """
    t = np.asarray(t, dtype=float)
    a = np.asarray(a, dtype=float)
    ss_res = np.sum((t - a) ** 2)
    ss_tot = np.sum((t - np.mean(t)) ** 2)
    return float(1.0 - ss_res / ss_tot)


def avaliar(t, a):
    """Calcula todas as métricas e devolve um dicionário."""
    return {
        "MSE": mse(t, a),
        "RMSE": rmse(t, a),
        "MAE": mae(t, a),
        "R2": r2(t, a),
    }
