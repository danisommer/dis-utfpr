"""
CGNE — Conjugate Gradient Normal Error.

Algoritmo:
    f0 = 0
    r0 = g - H * f0
    p0 = H^T * r0
    while not convergiu:
        alpha = (r_i^T * r_i) / (p_i^T * p_i)
        f     = f + alpha * p_i
        r     = r - alpha * H * p_i
        beta  = (r_i+1^T * r_i+1) / (r_i^T * r_i)
        p_i+1 = H^T * r_i+1 + beta * p_i
"""

from __future__ import annotations

import time
from typing import Tuple

import numpy as np


def cgne(
    H: np.ndarray,
    g: np.ndarray,
    max_iter: int = 10,
    tol: float = 1e-4,
) -> Tuple[np.ndarray, int, float]:
    """Reconstrucao por Conjugate Gradient Normal Error.

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
    p = H.T @ r

    r_norm_sq = float(r @ r)
    prev_r_norm = float(np.sqrt(r_norm_sq))

    n_iter = 0
    for i in range(max_iter):
        n_iter = i + 1

        p_norm_sq = float(p @ p)
        if p_norm_sq == 0.0:
            break

        alpha = r_norm_sq / p_norm_sq

        f = f + alpha * p
        r = r - alpha * (H @ p)

        new_r_norm_sq = float(r @ r)
        new_r_norm = float(np.sqrt(new_r_norm_sq))

        epsilon = new_r_norm - prev_r_norm
        if abs(epsilon) < tol:
            break
        prev_r_norm = new_r_norm

        if r_norm_sq == 0.0:
            break
        beta = new_r_norm_sq / r_norm_sq

        p = H.T @ r + beta * p
        r_norm_sq = new_r_norm_sq

    tempo_total = time.perf_counter() - t0
    return f, n_iter, tempo_total
