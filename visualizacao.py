"""
Visualização dos resultados do ADALINE na base Auto MPG.

Gráficos:
    1. Valores reais x valores previstos (conjunto de teste)
    2. Resíduos e = t - a x valores previstos (conjunto de teste)
    3. Evolução do MSE ao final de cada época (treinamento e teste)

Notação do livro: t é o consumo real (y_test) e a é a saída da rede, o consumo
previsto (y_pred). O resíduo e_i = y_i - y_hat_i é exatamente o erro e = t - a
do algoritmo LMS.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker

# Cores da paleta de referência (tema claro).
SUPERFICIE = "#fcfcfb"
TEXTO_PRIMARIO = "#0b0b0b"
TEXTO_SECUNDARIO = "#52514e"
TEXTO_SUAVE = "#898781"
GRADE = "#e1e0d9"
EIXO = "#c3c2b7"
SERIE_1 = "#2a78d6"  # azul: treinamento / pontos
SERIE_2 = "#eb6834"  # laranja: teste / tendência

ESTILO = {
    "figure.facecolor": SUPERFICIE,
    "axes.facecolor": SUPERFICIE,
    "savefig.facecolor": SUPERFICIE,
    "axes.edgecolor": EIXO,
    "axes.linewidth": 1.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRADE,
    "grid.linewidth": 0.8,
    "grid.linestyle": "-",
    "axes.titlesize": 12,
    "axes.titleweight": "semibold",
    "axes.titlelocation": "left",
    "axes.titlecolor": TEXTO_PRIMARIO,
    "axes.labelcolor": TEXTO_SECUNDARIO,
    "axes.labelsize": 10,
    "xtick.color": TEXTO_SUAVE,
    "ytick.color": TEXTO_SUAVE,
    "xtick.labelcolor": TEXTO_SECUNDARIO,
    "ytick.labelcolor": TEXTO_SECUNDARIO,
    "text.color": TEXTO_PRIMARIO,
    "font.size": 10,
    "legend.frameon": False,
    "legend.fontsize": 9,
    "lines.linewidth": 2,
    "lines.solid_capstyle": "round",
    "lines.solid_joinstyle": "round",
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
}

# Pontos com anel na cor da superfície para continuarem legíveis quando se sobrepõem.
PONTOS = {"s": 36, "color": SERIE_1, "edgecolors": SUPERFICIE, "linewidths": 1.0, "alpha": 0.9, "zorder": 3}


def _texto_metricas(metricas):
    return "\n".join([
        f"MSE = {metricas['MSE']:.3f} mpg²",
        f"RMSE = {metricas['RMSE']:.3f} mpg",
        f"MAE = {metricas['MAE']:.3f} mpg",
        f"R² = {metricas['R2']:.3f}",
    ])


def grafico_real_vs_previsto(t, a, metricas=None):
    """
    Consumo real t (y_test) no eixo x e consumo previsto a (y_pred) no eixo y.
    Cada ponto é um automóvel do conjunto de teste; a diagonal y = x representa
    previsões perfeitas.
    """
    t = np.ravel(t)
    a = np.ravel(a)

    with plt.rc_context(ESTILO): # type: ignore
        fig, ax = plt.subplots(figsize=(6.4, 6.0))

        minimo = min(t.min(), a.min())
        maximo = max(t.max(), a.max())
        margem = 0.05 * (maximo - minimo)
        limites = (minimo - margem, maximo + margem)

        ax.plot(limites, limites, color=TEXTO_SECUNDARIO, linewidth=1.2, zorder=2,
                label="$y = x$ (previsão perfeita)")
        ax.scatter(t, a, label=f"Automóvel do conjunto de teste (Q = {t.size})", **PONTOS)

        ax.set_xlim(limites)
        ax.set_ylim(limites)
        ax.set_aspect("equal")
        ax.set_xlabel("Consumo real, $t$ (mpg)")
        ax.set_ylabel("Consumo previsto pela rede, $a$ (mpg)")
        ax.set_title("Valores reais × valores previstos (teste)")
        ax.legend(loc="lower right")

        if metricas is not None:
            ax.text(0.03, 0.97, _texto_metricas(metricas), transform=ax.transAxes,
                    va="top", ha="left", color=TEXTO_SECUNDARIO, fontsize=9,
                    bbox={"boxstyle": "round,pad=0.5", "facecolor": SUPERFICIE, "edgecolor": GRADE})

        fig.tight_layout()
    return fig


def _suavizar(x, y, pontos=200, fracao_banda=0.12):
    """Média local com núcleo gaussiano (Nadaraya-Watson) para evidenciar padrões."""
    grade = np.linspace(x.min(), x.max(), pontos)
    banda = fracao_banda * (x.max() - x.min())
    pesos = np.exp(-0.5 * ((grade[:, None] - x[None, :]) / banda) ** 2)
    return grade, (pesos @ y) / pesos.sum(axis=1)


def grafico_residuos(t, a, nomes=None, n_destaques=3):
    """
    Resíduos e = t - a em função do consumo previsto a.

    O ideal é uma nuvem aleatória ao redor de zero. A curva de tendência (média
    local dos resíduos) ajuda a perceber curvaturas; um "funil" que se abre
    conforme a previsão cresce indica variância não constante.
    """
    t = np.ravel(t)
    a = np.ravel(a)
    e = t - a

    with plt.rc_context(ESTILO): # type: ignore
        fig, ax = plt.subplots(figsize=(8.0, 5.0))

        ax.axhline(0, color=TEXTO_SECUNDARIO, linewidth=1.2, zorder=2, label="Resíduo nulo ($e = 0$)")
        ax.scatter(a, e, label="Automóvel do conjunto de teste", **PONTOS)
        grade, tendencia = _suavizar(a, e)
        ax.plot(grade, tendencia, color=SERIE_2, zorder=4, label="Tendência média dos resíduos")

        if nomes is not None and n_destaques > 0:
            for i in np.argsort(-np.abs(e))[:n_destaques]:
                ax.annotate(nomes[i], (a[i], e[i]), xytext=(6, 4 if e[i] >= 0 else -10),
                            textcoords="offset points", fontsize=8, color=TEXTO_SECUNDARIO)

        limite = 1.15 * np.abs(e).max()
        ax.set_ylim(-limite, limite)
        ax.set_xlabel("Consumo previsto pela rede, $a$ (mpg)")
        ax.set_ylabel("Resíduo, $e = t - a$ (mpg)")
        ax.set_title("Resíduos × valores previstos (teste)")
        ax.legend(loc="lower left")
        fig.tight_layout()
    return fig


def grafico_evolucao_mse(historico, mse_minimo=None, alpha=None):
    """
    MSE ao final de cada época para os conjuntos de treinamento e teste.

    O painel da esquerda mostra todas as épocas em escala logarítmica (a queda
    inicial é de uma ordem de grandeza); o da direita amplia as épocas finais
    em escala linear, onde aparecem oscilações e a eventual piora do teste.
    """
    mse_treino = np.asarray(historico["mse_treino"])
    mse_teste = np.asarray(historico["mse_teste"])
    epocas = np.arange(1, mse_treino.size + 1)
    tem_teste = mse_teste.size == mse_treino.size
    inicio_detalhe = max(1, mse_treino.size // 10)

    with plt.rc_context(ESTILO): # type: ignore
        fig, (ax_total, ax_detalhe) = plt.subplots(1, 2, figsize=(12.0, 4.8))

        for ax, fatia in ((ax_total, slice(None)), (ax_detalhe, slice(inicio_detalhe - 1, None))):
            if mse_minimo is not None:
                ax.axhline(mse_minimo, color=TEXTO_SUAVE, linewidth=1.0, zorder=1,
                           label=r"Mínimo de $F(\mathbf{x})$ no treino, $\mathbf{x}^* = \mathbf{R}^{-1}\mathbf{h}$")
            ax.plot(epocas[fatia], mse_treino[fatia], color=SERIE_1, zorder=3, label="Treinamento")
            if tem_teste:
                ax.plot(epocas[fatia], mse_teste[fatia], color=SERIE_2, zorder=3, label="Teste")
            ax.set_xlabel("Época")
            ax.set_ylabel("MSE (mpg²)")
            ax.set_xlim(epocas[fatia][0], epocas[fatia][-1])

        valores_total = [mse_treino] + ([mse_teste] if tem_teste else []) + ([[mse_minimo]] if mse_minimo is not None else [])
        valores_total = np.concatenate(valores_total)
        ax_total.set_yscale("log")
        ax_total.set_ylim(valores_total.min() / 1.12, valores_total.max() * 1.08)
        ax_total.yaxis.set_major_locator(ticker.LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
        ax_total.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax_total.yaxis.set_minor_formatter(ticker.NullFormatter())
        ax_total.set_title("Todas as épocas (escala log)")
        ax_total.legend(loc="upper right")

        ax_detalhe.set_title(f"Detalhe a partir da época {inicio_detalhe}")
        valores_detalhe = [mse_treino[inicio_detalhe - 1:]]
        if tem_teste:
            valores_detalhe.append(mse_teste[inicio_detalhe - 1:])
        if mse_minimo is not None:
            valores_detalhe.append(np.array([mse_minimo]))
        valores_detalhe = np.concatenate(valores_detalhe)
        folga = 0.08 * (valores_detalhe.max() - valores_detalhe.min())
        ax_detalhe.set_ylim(valores_detalhe.min() - 2 * folga, valores_detalhe.max() + folga)

        if tem_teste:
            # Destaca a época de menor MSE de teste (início de uma possível piora da generalização).
            melhor = int(np.argmin(mse_teste))
            ax = ax_detalhe if epocas[melhor] >= inicio_detalhe else ax_total
            a_direita = epocas[melhor] > 0.7 * epocas[-1]
            ax.scatter([epocas[melhor]], [mse_teste[melhor]], s=48, color=SERIE_2,
                       edgecolors=SUPERFICIE, linewidths=2, zorder=5)
            ax.annotate(f"menor MSE de teste: {mse_teste[melhor]:.3f} (época {epocas[melhor]})",
                        (epocas[melhor], mse_teste[melhor]), xytext=(-8 if a_direita else 8, -12),
                        textcoords="offset points", ha="right" if a_direita else "left",
                        fontsize=8, color=TEXTO_SECUNDARIO)

        titulo = "Evolução do MSE durante o treinamento (LMS"
        titulo += f", $\\alpha$ = {alpha:g})" if alpha is not None else ")"
        fig.suptitle(titulo, x=0.01, ha="left", fontsize=13, fontweight="semibold", color=TEXTO_PRIMARIO)
        fig.tight_layout()
    return fig


def gerar_graficos(t_teste, a_teste, historico, metricas_teste=None, mse_minimo=None,
                   alpha=None, nomes_teste=None, pasta_saida="resultados", mostrar=True):
    """Gera os três gráficos, salva-os como PNG em pasta_saida e, se pedido, exibe-os."""
    pasta_saida = Path(pasta_saida)
    pasta_saida.mkdir(parents=True, exist_ok=True)

    figuras = {
        "1_reais_vs_previstos.png": grafico_real_vs_previsto(t_teste, a_teste, metricas_teste),
        "2_residuos.png": grafico_residuos(t_teste, a_teste, nomes_teste),
        "3_evolucao_mse.png": grafico_evolucao_mse(historico, mse_minimo, alpha),
    }

    caminhos = []
    for nome, figura in figuras.items():
        caminho = pasta_saida / nome
        figura.savefig(caminho)
        caminhos.append(caminho)

    if mostrar:
        plt.show()
    else:
        for figura in figuras.values():
            plt.close(figura)
    return caminhos
