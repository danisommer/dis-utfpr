"""
Servidor HTTP interpretado (Flask) — porta 5001.

Endpoint:
    POST /reconstruct
        Recebe JSON:
            {
                "g": [...],             # vetor de sinal (lista de floats)
                "H": [[...], ...],      # opcional — matriz de modelo
                "H_path": "...",        # opcional — caminho local p/ matriz cacheada
                "algorithm": "cgnr"|"cgne",
                "model": 1 | 2,
                "apply_gain": true|false
            }
        Responde JSON:
            {
                "algorithm": ...,
                "image_base64": "<PNG em base64>",
                "width": int, "height": int,
                "n_iter": int,
                "tempo_reconstrucao_s": float,
                "started_at": "ISO8601",
                "finished_at": "ISO8601",
                "server": "python"
            }
"""

from __future__ import annotations

import base64
import io
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Optional

import numpy as np
from flask import Flask, jsonify, request
from PIL import Image, PngImagePlugin

from cgne import cgne
from cgnr import cgnr
from signal_gain import apply_signal_gain

LOG = logging.getLogger("server-interpreted")
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
)

app = Flask(__name__)

MODEL_CONFIG = {
    1: {"S": 794, "N": 64, "size": (60, 60)},
    2: {"S": 436, "N": 64, "size": (30, 30)},
}

H_CACHE: dict[str, np.ndarray] = {}


def _load_H(path: str) -> np.ndarray:
    """Carrega matriz H de arquivo .npy ou texto (com cache em memoria)."""
    if path in H_CACHE:
        return H_CACHE[path]
    if not os.path.exists(path):
        raise FileNotFoundError(f"Matriz H nao encontrada: {path}")

    if path.endswith(".npy"):
        H = np.load(path)
    elif path.endswith(".csv"):
        H = np.loadtxt(path, delimiter=",", dtype=np.float64)
    else:
        H = np.loadtxt(path, dtype=np.float64)

    H_CACHE[path] = H
    LOG.info("Matriz H carregada de %s, shape=%s", path, H.shape)
    return H


def _vector_to_png(
    f: np.ndarray, width: int, height: int, metadata: dict
) -> bytes:
    """Converte um vetor reconstruido em um PNG (com metadados tEXt)."""
    arr = np.asarray(f, dtype=np.float64).reshape((height, width), order="F")
    arr = arr - arr.min()
    if arr.max() > 0:
        arr = arr / arr.max()
    arr = (arr * 255.0).clip(0, 255).astype(np.uint8)

    img = Image.fromarray(arr, mode="L")

    info = PngImagePlugin.PngInfo()
    for k, v in metadata.items():
        info.add_text(str(k), str(v))

    buf = io.BytesIO()
    img.save(buf, format="PNG", pnginfo=info)
    return buf.getvalue()


@app.get("/health")
def health() -> tuple:
    return jsonify({"status": "ok", "server": "python"}), 200


@app.post("/reconstruct")
def reconstruct() -> tuple:
    payload = request.get_json(force=True, silent=False)

    algorithm: str = str(payload.get("algorithm", "cgnr")).lower()
    model: int = int(payload.get("model", 1))
    apply_gain: bool = bool(payload.get("apply_gain", True))

    if model not in MODEL_CONFIG:
        return jsonify({"error": f"modelo invalido: {model}"}), 400
    cfg = MODEL_CONFIG[model]
    S, N = cfg["S"], cfg["N"]
    width, height = cfg["size"]

    g_raw = payload.get("g")
    if g_raw is None:
        return jsonify({"error": "campo 'g' ausente"}), 400
    g = np.asarray(g_raw, dtype=np.float64).ravel()

    H: Optional[np.ndarray] = None
    if "H" in payload and payload["H"] is not None:
        H = np.asarray(payload["H"], dtype=np.float64)
    elif "H_path" in payload and payload["H_path"]:
        H = _load_H(payload["H_path"])
    else:
        default_path = os.environ.get(
            f"H_MODEL_{model}_PATH",
            os.path.join("data", f"H_modelo_{model}.npy"),
        )
        H = _load_H(default_path)

    if apply_gain:
        try:
            g = apply_signal_gain(g, S=S, N=N)
        except Exception as exc:
            LOG.warning("Falha no ganho de sinal: %s", exc)

    started_at = datetime.now(timezone.utc)

    if algorithm == "cgnr":
        f, n_iter, tempo = cgnr(H, g)
    elif algorithm == "cgne":
        f, n_iter, tempo = cgne(H, g)
    else:
        return jsonify({"error": f"algoritmo invalido: {algorithm}"}), 400

    finished_at = datetime.now(timezone.utc)

    metadata = {
        "algorithm": algorithm.upper(),
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "size": f"{width}x{height}",
        "iterations": n_iter,
        "server": "python",
    }

    png_bytes = _vector_to_png(f, width, height, metadata)
    image_b64 = base64.b64encode(png_bytes).decode("ascii")

    LOG.info(
        "reconstruct ok algo=%s model=%d iter=%d tempo=%.4fs",
        algorithm,
        model,
        n_iter,
        tempo,
    )

    return (
        jsonify(
            {
                "algorithm": algorithm.upper(),
                "image_base64": image_b64,
                "width": width,
                "height": height,
                "n_iter": n_iter,
                "tempo_reconstrucao_s": tempo,
                "started_at": started_at.isoformat(),
                "finished_at": finished_at.isoformat(),
                "server": "python",
            }
        ),
        200,
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    LOG.info("Servidor interpretado iniciando na porta %d", port)
    try:
        app.run(host="0.0.0.0", port=port, threaded=True)
    except KeyboardInterrupt:
        LOG.info("Servidor encerrado pelo usuario.")
        sys.exit(0)
