"""
CGNR — Conjugate Gradient Normal Residual.

Implementacao do algoritmo iterativo para resolver sistemas H * f = g
no sentido de minimos quadrados, em problemas mal condicionados de
reconstrucao de imagens.

Algoritmo:
    f0 = 0
    r0 = g - H * f0
    z0 = H^T * r0
    p0 = z0
    while not convergiu:
        w_i   = H * p_i
        alpha = ||z_i||^2 / ||w_i||^2
        f     = f + alpha * p_i
        r     = r - alpha * w_i
        z_i+1 = H^T * r
        beta  = ||z_i+1||^2 / ||z_i||^2
        p_i+1 = z_i+1 + beta * p_i
"""

from __future__ import annotations

import time
from typing import Tuple

import numpy as np


def cgnr(
    H: np.ndarray,
    g: np.ndarray,
    max_iter: int = 10,
    tol: float = 1e-4,
) -> Tuple[np.ndarray, int, float]:
    """Reconstrucao por Conjugate Gradient Normal Residual.

    Args:
        H: matriz de modelo, shape (S, M).
        g: vetor de sinal, shape (S,).
        max_iter: numero maximo de iteracoes (default 10).
        tol: tolerancia para o criterio de parada |epsilon| (default 1e-4).

    Returns:
        f: vetor da imagem reconstruida, shape (M,).
        n_iter: numero de iteracoes efetivamente executadas.
        tempo_total: duracao da reconstrucao em segundos.
    """
    t0 = time.perf_counter()

    H = np.asarray(H, dtype=np.float64)
    g = np.asarray(g, dtype=np.float64).ravel()

    m = H.shape[1]

    f = np.zeros(m, dtype=np.float64)
    r = g - H @ f
    z = H.T @ r
    p = z.copy()

    z_norm_sq = float(z @ z)
    # epsilon = ||r_i+1||_2 - ||r_i||_2 (diferenca de normas, conforme enunciado)
    prev_r_norm = float(np.sqrt(r @ r))

    n_iter = 0
    for i in range(max_iter):
        n_iter = i + 1

        w = H @ p
        w_norm_sq = float(w @ w)
        if w_norm_sq == 0.0:
            break

        alpha = z_norm_sq / w_norm_sq

        f = f + alpha * p
        r = r - alpha * w

        new_r_norm = float(np.sqrt(r @ r))
        epsilon = new_r_norm - prev_r_norm
        if abs(epsilon) < tol:
            break
        prev_r_norm = new_r_norm

        z_next = H.T @ r
        z_next_norm_sq = float(z_next @ z_next)

        if z_norm_sq == 0.0:
            break

        beta = z_next_norm_sq / z_norm_sq

        p = z_next + beta * p
        z = z_next
        z_norm_sq = z_next_norm_sq

    tempo_total = time.perf_counter() - t0
    return f, n_iter, tempo_total
