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
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

import numpy as np
import requests

from report_generator import ReconstructionResult, generate_report


@dataclass
class SignalFile:
    """Sinal de teste pronto para envio."""

    path: str
    apply_gain: bool  # True se o sinal e bruto e precisa de ganho

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


def _discover_signals(data_dir: str, model: int) -> List[SignalFile]:
    """Descobre sinais de teste para o modelo dado.

    Convencao dos arquivos do professor:
        Modelo 1 (60x60):
            - G-*.csv          -> sinais brutos          (apply_gain=True)
            - A-60x60-*.csv    -> sinais com ganho ja aplicado (apply_gain=False)
        Modelo 2 (30x30):
            - g-30x30-*.csv    -> sinais brutos          (apply_gain=True)
            - A-30x30-*.csv    -> sinais com ganho ja aplicado (apply_gain=False)

    Tambem aceita .npy nas mesmas nomenclaturas para quem pre-converter.
    """
    if not os.path.isdir(data_dir):
        return []

    exts = (".csv", ".npy", ".txt")
    files = sorted(os.listdir(data_dir))

    def _is_signal(fn: str) -> bool:
        return fn.lower().endswith(exts)

    found: List[SignalFile] = []

    if model == 1:
        for fn in files:
            if not _is_signal(fn):
                continue
            low = fn.lower()
            if low.startswith("g-") and not low.startswith("g-30x30"):
                found.append(SignalFile(os.path.join(data_dir, fn), apply_gain=True))
            elif low.startswith("a-60x60"):
                found.append(SignalFile(os.path.join(data_dir, fn), apply_gain=False))
    elif model == 2:
        for fn in files:
            if not _is_signal(fn):
                continue
            low = fn.lower()
            if low.startswith("g-30x30"):
                found.append(SignalFile(os.path.join(data_dir, fn), apply_gain=True))
            elif low.startswith("a-30x30"):
                found.append(SignalFile(os.path.join(data_dir, fn), apply_gain=False))
    return found


def _resolve_h_path(data_dir: str, model: int) -> Optional[str]:
    """Resolve o caminho do arquivo H para o modelo (padroes do professor).

    Procura, em ordem: H-<model>.npy (mais rapido), H-<model>.csv,
    e os nomes alternativos H_modelo_<model>.npy/.csv.
    """
    candidates = [
        f"H-{model}.npy",
        f"H-{model}.csv",
        f"H_modelo_{model}.npy",
        f"H_modelo_{model}.csv",
    ]
    for name in candidates:
        p = os.path.join(data_dir, name)
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
    apply_gain: bool,
) -> Optional[ReconstructionResult]:
    """Envia uma requisicao para um servidor e devolve o resultado."""
    payload = {
        "g": g.tolist(),
        "algorithm": algorithm,
        "model": model,
        "apply_gain": apply_gain,
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

            signal = rng.choice(signals)
            try:
                g = _load_signal(signal.path)
            except Exception as exc:  # noqa: BLE001
                LOG.error("[%s] erro lendo %s: %s", request_id, signal.path, exc)
                continue

            h_path = _resolve_h_path(data_dir, model)
            if h_path is None:
                LOG.warning(
                    "[%s] matriz H nao encontrada para modelo %d em %s — pulando",
                    request_id,
                    model,
                    data_dir,
                )
                continue

            LOG.info(
                "[%s] rodada %d/%d modelo=%d algo=%s sinal=%s ganho=%s",
                request_id,
                round_idx,
                rounds,
                model,
                algorithm,
                os.path.basename(signal.path),
                "sim" if signal.apply_gain else "ja_aplicado",
            )

            futs = {
                pool.submit(
                    _send_one,
                    name,
                    url,
                    g,
                    algorithm,
                    model,
                    h_path,
                    request_id,
                    timeout_s,
                    signal.apply_gain,
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
