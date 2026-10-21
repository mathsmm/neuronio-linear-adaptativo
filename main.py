"""
Rotina principal: treina um ADALINE com o algoritmo LMS para prever o consumo
(mpg) da base Auto MPG, avalia o modelo e gera os gráficos.

Uso:
    python main.py
    python main.py --alpha 0.01 --epocas 300
    python main.py --sem-janela        (apenas salva os gráficos em resultados/)
"""
import argparse
import sys
from pathlib import Path

import numpy as np

from adaline import ADALINE, solucao_minimo_mse, taxa_aprendizagem_maxima
from dados import preparar_dados
from metricas import avaliar
from visualizacao import gerar_graficos

PASTA = Path(__file__).resolve().parent
ARQUIVO_DADOS = PASTA / "auto-mpg.data"
PASTA_RESULTADOS = PASTA / "resultados"


def ler_argumentos():
    parser = argparse.ArgumentParser(description="ADALINE + LMS na base Auto MPG")
    parser.add_argument("--alpha", type=float, default=0.001, help="taxa de aprendizagem (padrão: 0.001)")
    parser.add_argument("--epocas", type=int, default=200, help="número de épocas (padrão: 200)")
    parser.add_argument("--fracao-teste", type=float, default=0.2, help="fração dos dados para teste (padrão: 0.2)")
    parser.add_argument("--semente", type=int, default=42, help="semente da separação e do embaralhamento")
    parser.add_argument("--inicializacao", choices=["zeros", "aleatoria"], default="zeros",
                        help="pesos iniciais (padrão: zeros, como no livro)")
    parser.add_argument("--sem-janela", action="store_true", help="não abre as janelas dos gráficos")
    return parser.parse_args()


def imprimir_metricas(titulo, metricas):
    print(f"  {titulo:<10} MSE = {metricas['MSE']:8.4f}   RMSE = {metricas['RMSE']:7.4f}   "
          f"MAE = {metricas['MAE']:7.4f}   R² = {metricas['R2']:7.4f}")


def main():
    # Evita caracteres corrompidos quando a saída é redirecionada no Windows.
    sys.stdout.reconfigure(encoding="utf-8")
    args = ler_argumentos()

    dados = preparar_dados(ARQUIVO_DADOS, fracao_teste=args.fracao_teste, semente=args.semente)
    P_treino, T_treino = dados["P_treino"], dados["T_treino"]
    P_teste, T_teste = dados["P_teste"], dados["T_teste"]
    R, Q_treino = P_treino.shape

    print("Base Auto MPG")
    print(f"  Qtd. variáveis (R) = {R}: {', '.join(dados['nomes_entradas'])}")
    print(f"  {Q_treino} entradas p/ treinamento; {P_teste.shape[1]} entradas p/ teste")

    # Limite de estabilidade do LMS (Eq. 10.44): 0 < alpha < 1 / lambda_max.
    alpha_max, autovalores = taxa_aprendizagem_maxima(P_treino)
    print("\nMatriz de correlação R = E[z z^T]")
    print(f"  Autovalores: {np.array2string(autovalores, precision=4)}")
    print(f"  lambda_max = {autovalores.max():.4f}  ->  alpha < 1/lambda_max = {alpha_max:.4f}")
    print(f"  alpha escolhido = {args.alpha:g}")
    if args.alpha >= alpha_max:
        print("  ATENÇÃO: alpha fora do limite de estabilidade; o LMS deve divergir.")

    rede = ADALINE(R=R, S=1, alpha=args.alpha, inicializacao=args.inicializacao, semente=args.semente)
    historico = rede.treinar(P_treino, T_treino, args.epocas, P_teste, T_teste)
    print(f"\nTreinamento LMS: {args.epocas} épocas, {rede.k} iterações k")

    # Comparação com o ponto de mínimo do MSE, x* = R^-1 h (Eq. 10.18).
    x_otimo, F_otimo = solucao_minimo_mse(P_treino, T_treino)
    x_lms = rede.vetor_x()
    print("\nParâmetros x = [1w; b] (entradas padronizadas)")
    print(f"  {'parâmetro':<16}{'LMS':>10}{'x* = R^-1 h':>14}")
    for nome, valor_lms, valor_otimo in zip(dados["nomes_entradas"] + ["b (bias)"], x_lms, x_otimo):
        print(f"  {nome:<16}{valor_lms:>10.4f}{valor_otimo:>14.4f}")
    print(f"  ||x_LMS - x*|| = {np.linalg.norm(x_lms - x_otimo):.4f}")
    print(f"  F(x*) no treinamento = {F_otimo:.4f}   MSE final do LMS no treinamento = {historico['mse_treino'][-1]:.4f}")

    a_treino = rede.simular(P_treino)
    a_teste = rede.simular(P_teste)
    metricas_treino = avaliar(T_treino, a_treino)
    metricas_teste = avaliar(T_teste, a_teste)

    print("\nAvaliação do ADALINE")
    imprimir_metricas("Treino", metricas_treino)
    imprimir_metricas("Teste", metricas_teste)

    caminhos = gerar_graficos(
        t_teste=T_teste.ravel(),
        a_teste=a_teste.ravel(),
        historico=historico,
        metricas_teste=metricas_teste,
        mse_minimo=F_otimo,
        alpha=args.alpha,
        nomes_teste=dados["carros_teste"],
        pasta_saida=PASTA_RESULTADOS,
        mostrar=not args.sem_janela,
    )
    print("\nGráficos salvos em:")
    for caminho in caminhos:
        print(f"  {caminho.relative_to(PASTA)}")


if __name__ == "__main__":
    main()
