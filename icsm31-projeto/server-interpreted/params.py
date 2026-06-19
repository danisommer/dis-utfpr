"""
Parametros definidos no enunciado (Algoritmos e definicoes).

    c = ||H^T * H||_2              # Fator de reducao
    lambda = max(abs(H^T * g)) * 0.10  # Coeficiente de regularizacao

Como ||H^T H||_2 = sigma_max(H)^2 (maior autovalor de H^T H), o fator de
reducao e obtido por iteracao de potencia sobre H^T H, evitando montar a
matriz cheia ou rodar uma SVD completa em H (50816 x 3600). O resultado e
cacheado por caminho de H, pois depende apenas da matriz de modelo.
"""

from __future__ import annotations

import numpy as np

# Cache do fator de reducao por chave (caminho de H). c depende so de H.
_C_CACHE: dict[str, float] = {}


def reduction_factor(
    H: np.ndarray,
    cache_key: str | None = None,
    max_iter: int = 200,
    tol: float = 1e-9,
) -> float:
    """Calcula c = ||H^T H||_2 (maior autovalor de H^T H) por iteracao de potencia.

    Usa um vetor inicial deterministico (todos 1) — identico ao do servidor Go —
    para que ambas as versoes produzam o mesmo c para a mesma matriz H.

    Args:
        H: matriz de modelo, shape (S, M).
        cache_key: chave de cache (ex.: caminho do arquivo de H). Se None, nao cacheia.
        max_iter: numero maximo de iteracoes de potencia.
        tol: tolerancia relativa para parada antecipada.

    Returns:
        c: o valor da norma-2 de H^T H (>= 0).
    """
    if cache_key is not None and cache_key in _C_CACHE:
        return _C_CACHE[cache_key]

    m = H.shape[1]
    v = np.ones(m, dtype=np.float64)
    nv = np.linalg.norm(v)
    if nv == 0.0:
        return 0.0
    v /= nv

    eigval = 0.0
    for _ in range(max_iter):
        # w = (H^T H) v, sem montar H^T H explicitamente
        w = H.T @ (H @ v)
        nw = float(np.linalg.norm(w))
        if nw == 0.0:
            eigval = 0.0
            break
        v = w / nw
        if abs(nw - eigval) <= tol * nw:
            eigval = nw
            break
        eigval = nw

    c = float(eigval)
    if cache_key is not None:
        _C_CACHE[cache_key] = c
    return c


def regularization_lambda(H: np.ndarray, g: np.ndarray) -> float:
    """Calcula lambda = max(abs(H^T * g)) * 0.10."""
    htg = H.T @ np.asarray(g, dtype=np.float64).ravel()
    return float(np.max(np.abs(htg)) * 0.10)
