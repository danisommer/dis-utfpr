"""
Ganho de sinal aplicado ao vetor (ou matriz) g antes da reconstrucao.

Formula:
    for c = 1 .. N:
        for l = 1 .. S:
            gamma_l = 100 + (1/20) * sqrt(l * l)
            g[l, c] = g[l, c] * gamma_l
"""

from __future__ import annotations

import numpy as np


def apply_signal_gain(g: np.ndarray, S: int, N: int) -> np.ndarray:
    """Aplica o ganho gamma_l a cada amostra l do sinal.

    Args:
        g: vetor de sinal (shape (S*N,)) ou matriz (shape (S, N)).
        S: numero de amostras por sensor.
        N: numero de sensores.

    Returns:
        novo array com o ganho aplicado (mesmo shape do input).
    """
    g = np.asarray(g, dtype=np.float64)

    l_indices = np.arange(1, S + 1, dtype=np.float64)
    gamma = 100.0 + (1.0 / 20.0) * np.sqrt(l_indices * l_indices)

    if g.ndim == 1:
        if g.size == S * N:
            g_mat = g.reshape(S, N, order="F")
            g_mat = g_mat * gamma[:, np.newaxis]
            return g_mat.reshape(-1, order="F")
        # caso 1D com tamanho S apenas
        return g * gamma[: g.size]

    # caso 2D (S, N)
    return g * gamma[:, np.newaxis]
