#!/usr/bin/env python3
"""Convert the markdown validation form into a signable PDF.

Why a dedicated converter and not pandoc/md-to-pdf?
  • We control the attestation layout precisely — each DocuSign field is a
    ~40 pt tall rectangle with a printed anchor token next to it so the PDF
    can be uploaded to DocuSign and auto-place fields via anchor discovery.
  • Pure Python (reportlab) means no system toolchain — no LaTeX, no
    headless Chrome, no wkhtmltopdf.

Install (if needed):
    uv sync --extra pdf
  or: uv pip install reportlab

Usage:
    python scripts/validation_tool_to_pdf.py \
        --input  reports/validation_tool_filled.md \
        --output reports/validation_tool_filled.pdf

Tested against the output of `hinaing-eval score` (three-tier form with
DocuSign anchor tokens embedded per scripts/../agentic_hinaing_eval/report.py).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


def _require_reportlab():
    try:
        import reportlab  # noqa: F401
    except Exception:
        print(
            "reportlab is not installed. Install with:\n"
            "    uv sync --extra pdf\n"
            "  or:\n"
            "    uv pip install reportlab\n",
            file=sys.stderr,
        )
        sys.exit(2)


def _parse_table(lines: list[str], start: int) -> tuple[list[list[str]], int]:
    """Parse a GitHub-flavored markdown pipe-table starting at ``start``.

    Returns (rows, next_index). First row is the header; a separator row
    (|---|---|) is consumed and discarded.
    """
    rows: list[list[str]] = []
    i = start
    while i < len(lines) and lines[i].lstrip().startswith("|"):
        raw = lines[i].strip()
        # skip separator rows
        if re.fullmatch(r"\|\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)+\|", raw):
            i += 1
            continue
        cells = [c.strip() for c in raw.strip("|").split("|")]
        rows.append(cells)
        i += 1
    return rows, i


def _strip_inline_markdown(text: str) -> str:
    # **bold**, *italic*, `code`
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<!\*)\*([^*\n]+?)\*(?!\*)", r"<i>\1</i>", text)
    text = re.sub(r"`([^`]+?)`", r'<font face="Courier">\1</font>', text)
    return text


def build_pdf(markdown_text: str, output_path: Path) -> None:
    _require_reportlab()
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import LETTER
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    pt = 1.0  # reportlab measures in points by default
    from reportlab.platypus import (
        HRFlowable,
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    styles = getSampleStyleSheet()
    body = ParagraphStyle(
        "body",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=10,
        leading=13,
        spaceAfter=4,
    )
    h1 = ParagraphStyle("h1", parent=styles["Heading1"], fontSize=16, spaceAfter=10)
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontSize=13, spaceAfter=6)
    h3 = ParagraphStyle("h3", parent=styles["Heading3"], fontSize=11, spaceAfter=4)
    note = ParagraphStyle(
        "note", parent=body, fontSize=9, textColor=colors.HexColor("#555555"), leading=12
    )
    anchor_style = ParagraphStyle(
        "anchor",
        parent=body,
        fontSize=8,
        textColor=colors.HexColor("#888888"),
        fontName="Courier",
    )
    field_label = ParagraphStyle(
        "field_label", parent=body, fontName="Helvetica-Bold", fontSize=11, spaceAfter=2
    )

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=LETTER,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
    )
    story: list = []

    # Detect attestation signing fields and render them with tall blank boxes.
    # Recognised pattern in the markdown:
    #   **Expert name** `/docusign_name/`
    #   ``` (followed by blank lines and a closing ```)
    field_pattern = re.compile(r"\*\*(.+?)\*\*\s+`(/docusign_[a-z_]+/)`\s*$")

    lines = markdown_text.splitlines()
    i = 0

    def flush_paragraph(buf: list[str]) -> None:
        text = " ".join(buf).strip()
        if not text:
            return
        story.append(Paragraph(_strip_inline_markdown(text), body))

    para_buf: list[str] = []

    while i < len(lines):
        line = lines[i]
        stripped = line.rstrip()

        if not stripped.strip():
            flush_paragraph(para_buf)
            para_buf = []
            story.append(Spacer(1, 4))
            i += 1
            continue

        # Signature field block.
        match = field_pattern.match(stripped)
        if match:
            flush_paragraph(para_buf)
            para_buf = []
            label, anchor = match.group(1), match.group(2)
            story.append(Paragraph(_strip_inline_markdown(label), field_label))
            story.append(Paragraph(anchor, anchor_style))
            # Tall blank box (~45pt) with bottom underline for wet-sign fallback.
            box = Table(
                [[""]],
                colWidths=[6.8 * inch],
                rowHeights=[45 * pt],
            )
            box.setStyle(
                TableStyle([
                    ("BOX", (0, 0), (-1, -1), 0.4, colors.HexColor("#CCCCCC")),
                    ("LINEBELOW", (0, 0), (-1, -1), 0.8, colors.black),
                    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FAFAFA")),
                ])
            )
            story.append(box)
            story.append(Spacer(1, 10))
            # Skip the ```…``` fence that follows in the markdown
            i += 1
            if i < len(lines) and lines[i].strip() == "```":
                i += 1
                while i < len(lines) and lines[i].strip() != "```":
                    i += 1
                if i < len(lines):
                    i += 1  # consume closing fence
            continue

        # Headings
        if stripped.startswith("# "):
            flush_paragraph(para_buf); para_buf = []
            story.append(Paragraph(_strip_inline_markdown(stripped[2:]), h1))
            i += 1
            continue
        if stripped.startswith("## "):
            flush_paragraph(para_buf); para_buf = []
            story.append(Spacer(1, 6))
            story.append(Paragraph(_strip_inline_markdown(stripped[3:]), h2))
            i += 1
            continue
        if stripped.startswith("### "):
            flush_paragraph(para_buf); para_buf = []
            story.append(Spacer(1, 4))
            story.append(Paragraph(_strip_inline_markdown(stripped[4:]), h3))
            i += 1
            continue

        # Blockquote (caveats)
        if stripped.lstrip().startswith(">"):
            flush_paragraph(para_buf); para_buf = []
            text = stripped.lstrip()[1:].strip()
            story.append(Paragraph(_strip_inline_markdown(text), note))
            i += 1
            continue

        # HR (---)
        if stripped.strip() == "---":
            flush_paragraph(para_buf); para_buf = []
            story.append(HRFlowable(width="100%", thickness=0.4, color=colors.HexColor("#BBBBBB")))
            story.append(Spacer(1, 4))
            i += 1
            continue

        # Table
        if stripped.lstrip().startswith("|") and stripped.rstrip().endswith("|"):
            flush_paragraph(para_buf); para_buf = []
            rows, next_i = _parse_table(lines, i)
            if rows:
                # wrap each cell as a Paragraph so tables can wrap
                cell_style = ParagraphStyle("cell", parent=body, fontSize=8, leading=10)
                header_style = ParagraphStyle(
                    "cell_h", parent=cell_style, fontName="Helvetica-Bold", textColor=colors.white
                )
                wrapped: list[list] = []
                for r, row in enumerate(rows):
                    wrapped.append(
                        [Paragraph(_strip_inline_markdown(c), header_style if r == 0 else cell_style) for c in row]
                    )
                table = Table(wrapped, repeatRows=1)
                table.setStyle(
                    TableStyle([
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#333333")),
                        ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
                        ("GRID",       (0, 0), (-1, -1), 0.25, colors.HexColor("#BBBBBB")),
                        ("VALIGN",     (0, 0), (-1, -1), "TOP"),
                        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F5F5F5")]),
                        ("FONTSIZE",   (0, 0), (-1, -1), 8),
                        ("LEFTPADDING", (0, 0), (-1, -1), 4),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                    ])
                )
                story.append(table)
                story.append(Spacer(1, 6))
            i = next_i
            continue

        # Unordered list item
        if stripped.lstrip().startswith(("- ", "* ")):
            flush_paragraph(para_buf); para_buf = []
            indent = len(stripped) - len(stripped.lstrip())
            text = stripped.lstrip()[2:]
            bullet_style = ParagraphStyle(
                "bullet", parent=body, leftIndent=12 + indent, bulletIndent=indent, bulletFontName="Symbol"
            )
            story.append(Paragraph("• " + _strip_inline_markdown(text), bullet_style))
            i += 1
            continue

        # Fenced code block — render as-is in monospace
        if stripped.strip().startswith("```"):
            flush_paragraph(para_buf); para_buf = []
            i += 1
            code_buf: list[str] = []
            while i < len(lines) and lines[i].strip() != "```":
                code_buf.append(lines[i])
                i += 1
            if i < len(lines):
                i += 1
            if any(line.strip() for line in code_buf):
                code_style = ParagraphStyle(
                    "code", parent=body, fontName="Courier", fontSize=8, leading=10
                )
                for cl in code_buf:
                    story.append(Paragraph(cl.replace(" ", "&nbsp;") or "&nbsp;", code_style))
                story.append(Spacer(1, 4))
            continue

        # Default: accumulate into a paragraph buffer.
        para_buf.append(stripped)
        i += 1

    flush_paragraph(para_buf)

    doc.build(story)


def main() -> None:
    p = argparse.ArgumentParser(description="Render the validation-tool markdown as a signable PDF.")
    p.add_argument("--input", type=Path, default=Path("reports/validation_tool_filled.md"))
    p.add_argument("--output", type=Path, default=Path("reports/validation_tool_filled.pdf"))
    args = p.parse_args()
    if not args.input.exists():
        print(f"Input not found: {args.input}", file=sys.stderr)
        sys.exit(1)
    markdown = args.input.read_text(encoding="utf-8")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    build_pdf(markdown, args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
