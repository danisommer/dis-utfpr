"""
Cliente Python — envia sinais g aos dois servidores e coleta resultados.

Comportamento:
    1. Carrega matrizes H e sinais g do diretorio data/.
    2. Sorteia: modelo (1 ou 2), algoritmo (cgnr/cgne) e sinal g.
    3. Envia (em paralelo) para os dois servidores (Python:5001 e Go:5002).
    4. Espera um intervalo aleatorio (0.5 s a 3 s) entre rodadas.
    5. Apos N rodadas, gera relatorio PDF em reports/.

Uso:
    python client/client.py [--rounds 5] [--data-dir ./data] [--report-dir ./reports]
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures as futures
import logging
import os
import random
import sys
import time
import uuid
from datetime import datetime
from typing import List, Optional

import numpy as np
import requests

from report_generator import ReconstructionResult, generate_report

LOG = logging.getLogger("client")
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
)

SERVERS = {
    "python": "http://127.0.0.1:5001/reconstruct",
    "go": "http://127.0.0.1:5002/reconstruct",
}

MODEL_CONFIG = {
    1: {"S": 794, "N": 64, "size": (60, 60)},
    2: {"S": 436, "N": 64, "size": (30, 30)},
}


def _load_signal(path: str) -> np.ndarray:
    """Carrega um vetor de sinal g de .npy ou texto."""
    if path.endswith(".npy"):
        return np.load(path).astype(np.float64).ravel()
    return np.loadtxt(path, dtype=np.float64).ravel()


def _discover_signals(data_dir: str, model: int) -> List[str]:
    """Procura arquivos de sinal compativeis com o modelo dado.

    Convencoes aceitas:
        - data/sinais_modelo_<M>/*.npy
        - data/g_modelo_<M>_*.npy ou .csv
    """
    candidates: List[str] = []
    sub = os.path.join(data_dir, f"sinais_modelo_{model}")
    if os.path.isdir(sub):
        for fn in sorted(os.listdir(sub)):
            if fn.endswith((".npy", ".csv", ".txt")):
                candidates.append(os.path.join(sub, fn))
    for fn in sorted(os.listdir(data_dir)) if os.path.isdir(data_dir) else []:
        if fn.startswith(f"g_modelo_{model}") and fn.endswith((".npy", ".csv", ".txt")):
            candidates.append(os.path.join(data_dir, fn))
    return candidates


def _resolve_h_path(data_dir: str, model: int, prefer_csv: bool) -> Optional[str]:
    """Resolve o caminho do arquivo H apropriado para o servidor."""
    ext_pref = ("csv", "npy") if prefer_csv else ("npy", "csv")
    for ext in ext_pref:
        p = os.path.join(data_dir, f"H_modelo_{model}.{ext}")
        if os.path.exists(p):
            return os.path.abspath(p)
    return None


def _send_one(
    server_name: str,
    url: str,
    g: np.ndarray,
    algorithm: str,
    model: int,
    h_path: Optional[str],
    request_id: str,
    timeout_s: float,
) -> Optional[ReconstructionResult]:
    """Envia uma requisicao para um servidor e devolve o resultado."""
    payload = {
        "g": g.tolist(),
        "algorithm": algorithm,
        "model": model,
        "apply_gain": True,
    }
    if h_path:
        payload["H_path"] = h_path

    t0 = time.perf_counter()
    try:
        resp = requests.post(url, json=payload, timeout=timeout_s)
    except requests.RequestException as exc:
        LOG.error("[%s][%s] falha de rede: %s", request_id, server_name, exc)
        return None
    rtt = time.perf_counter() - t0

    if resp.status_code != 200:
        LOG.error(
            "[%s][%s] erro HTTP %d: %s",
            request_id,
            server_name,
            resp.status_code,
            resp.text[:200],
        )
        return None

    data = resp.json()
    LOG.info(
        "[%s][%s] algo=%s iter=%d tempo=%.4fs rtt=%.4fs",
        request_id,
        server_name,
        data.get("algorithm"),
        data.get("n_iter"),
        data.get("tempo_reconstrucao_s"),
        rtt,
    )

    try:
        img_bytes = base64.b64decode(data["image_base64"])
    except Exception as exc:  # noqa: BLE001
        LOG.error("[%s][%s] base64 invalido: %s", request_id, server_name, exc)
        return None

    return ReconstructionResult(
        server=str(data.get("server", server_name)),
        algorithm=str(data.get("algorithm", algorithm.upper())),
        width=int(data["width"]),
        height=int(data["height"]),
        n_iter=int(data["n_iter"]),
        tempo_reconstrucao_s=float(data["tempo_reconstrucao_s"]),
        started_at=str(data.get("started_at", "")),
        finished_at=str(data.get("finished_at", "")),
        image_png_bytes=img_bytes,
        model=model,
        request_id=request_id,
    )


def run_rounds(
    rounds: int,
    data_dir: str,
    report_dir: str,
    timeout_s: float,
    seed: Optional[int],
) -> None:
    rng = random.Random(seed)

    os.makedirs(report_dir, exist_ok=True)

    all_results: List[ReconstructionResult] = []

    with futures.ThreadPoolExecutor(max_workers=2) as pool:
        for round_idx in range(1, rounds + 1):
            model = rng.choice([1, 2])
            algorithm = rng.choice(["cgnr", "cgne"])
            request_id = uuid.uuid4().hex[:8]

            signals = _discover_signals(data_dir, model)
            if not signals:
                LOG.warning(
                    "[%s] sem sinais para modelo %d em %s — pulando rodada",
                    request_id,
                    model,
                    data_dir,
                )
                continue

            signal_path = rng.choice(signals)
            try:
                g = _load_signal(signal_path)
            except Exception as exc:  # noqa: BLE001
                LOG.error("[%s] erro lendo %s: %s", request_id, signal_path, exc)
                continue

            LOG.info(
                "[%s] rodada %d/%d modelo=%d algo=%s sinal=%s",
                request_id,
                round_idx,
                rounds,
                model,
                algorithm,
                os.path.basename(signal_path),
            )

            futs = {
                pool.submit(
                    _send_one,
                    name,
                    url,
                    g,
                    algorithm,
                    model,
                    _resolve_h_path(data_dir, model, prefer_csv=(name == "go")),
                    request_id,
                    timeout_s,
                ): name
                for name, url in SERVERS.items()
            }
            for fut in futures.as_completed(futs):
                rec = fut.result()
                if rec is not None:
                    all_results.append(rec)

            # intervalo aleatorio entre rodadas
            if round_idx < rounds:
                delay = rng.uniform(0.5, 3.0)
                LOG.info("[%s] aguardando %.2fs ate proxima rodada", request_id, delay)
                time.sleep(delay)

    if not all_results:
        LOG.warning("Nenhum resultado coletado; relatorio nao sera gerado.")
        return

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(report_dir, f"relatorio_{ts}.pdf")
    generate_report(all_results, out_path)
    LOG.info("Relatorio gerado em %s (%d reconstrucoes)", out_path, len(all_results))


def main() -> int:
    parser = argparse.ArgumentParser(description="Cliente ICSM31 — reconstrucao de imagens")
    parser.add_argument("--rounds", type=int, default=5, help="numero de rodadas (default 5)")
    parser.add_argument("--data-dir", default="data", help="diretorio dos dados (default ./data)")
    parser.add_argument(
        "--report-dir", default="reports", help="diretorio de saida (default ./reports)"
    )
    parser.add_argument("--timeout", type=float, default=300.0, help="timeout HTTP em segundos")
    parser.add_argument("--seed", type=int, default=None, help="seed do RNG (opcional)")
    args = parser.parse_args()

    if not os.path.isdir(args.data_dir):
        LOG.error("Diretorio de dados nao existe: %s", args.data_dir)
        return 1

    run_rounds(
        rounds=args.rounds,
        data_dir=args.data_dir,
        report_dir=args.report_dir,
        timeout_s=args.timeout,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
