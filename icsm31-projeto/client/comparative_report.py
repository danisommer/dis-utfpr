"""
Relatorio comparativo preenchido a partir das reconstrucoes coletadas.

Agrega os resultados das duas versoes de servidor (interpretado Python x
compilado Go) e produz um Markdown com tabelas de tempo medio, desvio padrao,
iteracoes medias e throughput, alem de informacoes do ambiente de execucao.

Diferente de `relatorio_comparativo_template.md` (em branco), este arquivo e
gerado automaticamente com os numeros reais de cada rodada.
"""

from __future__ import annotations

import platform
import statistics
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Tuple

from report_generator import ReconstructionResult

# Rotulo amigavel por servidor.
_SERVER_LABEL = {"python": "Python (interpretado)", "go": "Go (compilado)"}


def _fmt(value: float, nd: int = 2) -> str:
    return f"{value:.{nd}f}"


def _env_block() -> str:
    uname = platform.uname()
    return (
        "| Item | Valor |\n"
        "| --- | --- |\n"
        f"| Sistema operacional | {uname.system} {uname.release} |\n"
        f"| Arquitetura | {uname.machine} |\n"
        f"| Processador | {platform.processor() or uname.processor or 'n/d'} |\n"
        f"| Python | {platform.python_version()} |\n"
    )


def _aggregate(
    results: List[ReconstructionResult],
) -> Dict[Tuple[str, int, str], List[ReconstructionResult]]:
    """Agrupa por (algoritmo, modelo, servidor)."""
    groups: Dict[Tuple[str, int, str], List[ReconstructionResult]] = defaultdict(list)
    for r in results:
        groups[(r.algorithm.upper(), r.model, r.server.lower())].append(r)
    return groups


def _timings_table(results: List[ReconstructionResult]) -> str:
    groups = _aggregate(results)
    lines = [
        "| Algoritmo | Modelo | Servidor | Reconstrucoes | Tempo medio (ms) | Desvio (ms) | Iteracoes medias |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for key in sorted(groups.keys()):
        algo, model, server = key
        recs = groups[key]
        tempos_ms = [r.tempo_reconstrucao_s * 1000.0 for r in recs]
        iters = [r.n_iter for r in recs]
        media = statistics.mean(tempos_ms)
        desvio = statistics.pstdev(tempos_ms) if len(tempos_ms) > 1 else 0.0
        iter_med = statistics.mean(iters)
        label = _SERVER_LABEL.get(server, server)
        lines.append(
            f"| {algo} | {model} | {label} | {len(recs)} | "
            f"{_fmt(media)} | {_fmt(desvio)} | {_fmt(iter_med, 1)} |"
        )
    return "\n".join(lines)


def _throughput_table(results: List[ReconstructionResult]) -> str:
    by_server: Dict[str, List[ReconstructionResult]] = defaultdict(list)
    for r in results:
        by_server[r.server.lower()].append(r)

    lines = [
        "| Servidor | Reconstrucoes | Tempo total de CPU (s) | Tempo medio (ms) | Throughput (rec/s) |",
        "| --- | --- | --- | --- | --- |",
    ]
    for server in sorted(by_server.keys()):
        recs = by_server[server]
        total_s = sum(r.tempo_reconstrucao_s for r in recs)
        media_ms = (total_s / len(recs)) * 1000.0 if recs else 0.0
        throughput = (len(recs) / total_s) if total_s > 0 else 0.0
        label = _SERVER_LABEL.get(server, server)
        lines.append(
            f"| {label} | {len(recs)} | {_fmt(total_s, 4)} | "
            f"{_fmt(media_ms)} | {_fmt(throughput)} |"
        )
    return "\n".join(lines)


def _speedup_paragraph(results: List[ReconstructionResult]) -> str:
    by_server: Dict[str, List[float]] = defaultdict(list)
    for r in results:
        by_server[r.server.lower()].append(r.tempo_reconstrucao_s)

    if "python" in by_server and "go" in by_server:
        py = statistics.mean(by_server["python"])
        go = statistics.mean(by_server["go"])
        if go > 0:
            ratio = py / go
            faster = "Go (compilado)" if ratio >= 1 else "Python (interpretado)"
            factor = ratio if ratio >= 1 else (1.0 / ratio)
            return (
                f"Em media, **{faster}** foi **{_fmt(factor)}x** mais rapido "
                f"por reconstrucao (Python: {_fmt(py * 1000)} ms, "
                f"Go: {_fmt(go * 1000)} ms)."
            )
    return "_Dados insuficientes para comparar os dois servidores (rode com ambos ativos)._"


def generate_comparative_report(
    results: List[ReconstructionResult], output_path: str
) -> None:
    """Gera o relatorio comparativo (Markdown) com numeros reais."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    n_iter_total = sum(r.n_iter for r in results)

    md = f"""# Relatorio Comparativo (gerado automaticamente) — CGNR/CGNE: Python vs Go

**Disciplina:** ICSM31 — Desenvolvimento Integrado de Sistemas — UTFPR
**Gerado em:** {now}
**Total de reconstrucoes coletadas:** {len(results)}
**Total de iteracoes executadas:** {n_iter_total}

---

## 1. Ambiente de execucao

{_env_block()}

---

## 2. Tempos por (algoritmo x modelo x servidor)

{_timings_table(results)}

> Tempo de reconstrucao = campo `tempo_reconstrucao_s` retornado pelo servidor
> (nao inclui latencia de rede). Desvio = desvio padrao populacional.

---

## 3. Throughput por servidor

{_throughput_table(results)}

> Throughput = numero de reconstrucoes / soma dos tempos de reconstrucao do
> servidor. O objetivo do trabalho e maximizar reconstrucoes por unidade de tempo.

---

## 4. Sintese

{_speedup_paragraph(results)}

Observacoes:

- Ambos os servidores executam o **mesmo `g`** em cada rodada, garantindo
  comparacao justa entre as implementacoes.
- O numero de iteracoes tende a ser identico entre Python e Go para o mesmo
  par (algoritmo, sinal), pois implementam o mesmo metodo iterativo; pequenas
  diferencas, quando ocorrem, vem de arredondamento de ponto flutuante.
- O criterio de parada e `|epsilon| < 1e-4` ou 10 iteracoes (o que ocorrer primeiro).

---

_Relatorio gerado por `client/comparative_report.py`. Para a versao com analise
qualitativa das imagens e graficos, ver `reports/relatorio_comparativo_template.md`._
"""

    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(md)
