"""
Gerador de relatorio PDF com todas as imagens reconstruidas.

Cada pagina mostra:
    - Imagem reconstruida
    - Algoritmo (CGNR ou CGNE)
    - Servidor (Python ou Go)
    - Numero de iteracoes
    - Tempo de reconstrucao
    - Tamanho em pixels
    - Timestamps de inicio e fim
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import List

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image as RLImage,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


@dataclass
class ReconstructionResult:
    """Resultado de uma reconstrucao executada por um dos servidores."""

    server: str
    algorithm: str
    width: int
    height: int
    n_iter: int
    tempo_reconstrucao_s: float
    started_at: str
    finished_at: str
    image_png_bytes: bytes
    model: int
    request_id: str
    reduction_factor: float = 0.0
    lambda_reg: float = 0.0


def _render_table(rec: ReconstructionResult) -> Table:
    data = [
        ["Servidor", rec.server],
        ["Algoritmo", rec.algorithm],
        ["Modelo", str(rec.model)],
        ["Tamanho", f"{rec.width} x {rec.height}"],
        ["Iteracoes", str(rec.n_iter)],
        ["Tempo (s)", f"{rec.tempo_reconstrucao_s:.4f}"],
        ["c = ||H^T H||_2", f"{rec.reduction_factor:.6g}"],
        ["lambda", f"{rec.lambda_reg:.6g}"],
        ["Inicio", rec.started_at],
        ["Termino", rec.finished_at],
        ["Request ID", rec.request_id],
    ]
    table = Table(data, colWidths=[4 * cm, 11 * cm])
    table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("BACKGROUND", (0, 0), (0, -1), colors.lightgrey),
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )
    return table


def generate_report(
    results: List[ReconstructionResult], output_path: str
) -> None:
    """Gera um PDF em 'output_path' com todas as reconstrucoes."""
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
        title="Relatorio de Reconstrucoes — ICSM31",
    )
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("Relatorio de Reconstrucoes — CGNR / CGNE", styles["Title"]))
    story.append(Spacer(1, 0.4 * cm))
    story.append(
        Paragraph(
            f"Total de reconstrucoes: <b>{len(results)}</b>",
            styles["Normal"],
        )
    )
    story.append(Spacer(1, 0.6 * cm))

    for i, rec in enumerate(results, start=1):
        story.append(
            Paragraph(
                f"<b>#{i}</b> — {rec.algorithm} @ {rec.server}",
                styles["Heading3"],
            )
        )
        story.append(Spacer(1, 0.2 * cm))

        try:
            img_buf = io.BytesIO(rec.image_png_bytes)
            img = RLImage(img_buf, width=6 * cm, height=6 * cm, kind="proportional")
            story.append(img)
        except Exception as exc:  # noqa: BLE001
            story.append(Paragraph(f"<i>Falha ao renderizar imagem: {exc}</i>", styles["Italic"]))

        story.append(Spacer(1, 0.3 * cm))
        story.append(_render_table(rec))
        story.append(Spacer(1, 0.8 * cm))

    doc.build(story)
