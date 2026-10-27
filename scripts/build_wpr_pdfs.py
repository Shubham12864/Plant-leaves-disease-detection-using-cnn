#!/usr/bin/env python3
"""Render docs/WPR_*.md as PDFs for submission to the supervisor.

The PDFs are build output and are never committed. The week-commencing line is
left blank on purpose so the date can be written in by hand.

    python scripts/build_wpr_pdfs.py [--out DIR]
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import HRFlowable, Paragraph, SimpleDocTemplate, Spacer

ROOT = Path(__file__).resolve().parents[1]


def inline(text: str) -> str:
    """Markdown inline markup to reportlab markup."""
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"`([^`]+)`", r'<font face="Courier" size="9.5">\1</font>', text)
    return text


def build(md: Path, out: Path) -> None:
    base = getSampleStyleSheet()
    title = ParagraphStyle("t", parent=base["Title"], fontName="Times-Bold", fontSize=18, spaceAfter=10)
    head = ParagraphStyle("h", parent=base["Heading2"], fontName="Times-Bold", fontSize=13, spaceBefore=12, spaceAfter=4)
    body = ParagraphStyle("b", parent=base["BodyText"], fontName="Times-Roman", fontSize=11, leading=15, alignment=TA_JUSTIFY, spaceAfter=7)
    meta = ParagraphStyle("m", parent=body, alignment=0, spaceAfter=2)

    story: list = []
    paragraph: list[str] = []

    def flush() -> None:
        if paragraph:
            story.append(Paragraph(inline(" ".join(paragraph)), body))
            paragraph.clear()

    in_header = True
    for raw in md.read_text(encoding="utf-8").splitlines():
        line = raw.rstrip()
        if line.startswith("# "):
            flush()
            story.append(Paragraph(inline(line[2:]), title))
            story.append(HRFlowable(width="100%", thickness=0.8, color="#444444", spaceAfter=8))
        elif line.startswith("## "):
            flush()
            in_header = False
            story.append(Paragraph(inline(line[3:]), head))
        elif not line:
            flush()
        elif in_header and line.startswith("**"):
            flush()
            story.append(Paragraph(inline(line), meta))
        else:
            paragraph.append(line)
    flush()
    story.append(Spacer(1, 1.2 * cm))
    story.append(Paragraph("Submitted by: ______________________ &nbsp;&nbsp;&nbsp; Guide signature: ______________________", meta))

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Times-Roman", 9)
        canvas.drawCentredString(A4[0] / 2, 1.2 * cm, f"BT40 - {md.stem.replace('_', ' ')} - page {doc.page}")
        canvas.restoreState()

    doc = SimpleDocTemplate(
        str(out), pagesize=A4, leftMargin=2.4 * cm, rightMargin=2.4 * cm, topMargin=2.2 * cm, bottomMargin=2.2 * cm,
        title=f"BT40 {md.stem}", author="BT40 project team",
    )
    doc.build(story, onFirstPage=footer, onLaterPages=footer)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(Path.home() / "Downloads" / "BT40_submission"))
    args = parser.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for md in sorted((ROOT / "docs").glob("WPR_*.md")):
        target = out_dir / f"{md.stem}.pdf"
        build(md, target)
        print(f"built {target} ({target.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
