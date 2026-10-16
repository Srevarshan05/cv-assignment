"""Shared PDF report engine (plain academic style) for the Computer Vision assignments."""
import io
import os
from PIL import Image as PILImage, ImageDraw, ImageFont
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
                                PageBreak, KeepTogether, Preformatted, CondPageBreak)

LIB = "/usr/share/fonts/truetype/liberation"
pdfmetrics.registerFont(TTFont("Sans", f"{LIB}/LiberationSans-Regular.ttf"))
pdfmetrics.registerFont(TTFont("Sans-Bold", f"{LIB}/LiberationSans-Bold.ttf"))
pdfmetrics.registerFont(TTFont("Sans-Italic", f"{LIB}/LiberationSans-Italic.ttf"))
pdfmetrics.registerFont(TTFont("Mono", f"{LIB}/LiberationMono-Regular.ttf"))
pdfmetrics.registerFontFamily("Sans", normal="Sans", bold="Sans-Bold", italic="Sans-Italic", boldItalic="Sans-Bold")

STUDENT = [("Name", "Srevarshan"), ("Register Number", "RA2311026050039"), ("Year", "4th Year"),
           ("Department", "CSE - AIML"), ("Section", "A"), ("Subject", "Computer Vision"),
           ("Subject Code", "21CSE454T")]

PAGE_W, PAGE_H = A4
MARGIN = 22 * mm
CONTENT_W = PAGE_W - 2 * MARGIN

S = {
    "title": ParagraphStyle("title", fontName="Sans-Bold", fontSize=20, leading=26, alignment=TA_CENTER),
    "subtitle": ParagraphStyle("subtitle", fontName="Sans-Bold", fontSize=14, leading=20, alignment=TA_CENTER),
    "center": ParagraphStyle("center", fontName="Sans", fontSize=12, leading=17, alignment=TA_CENTER),
    "h1": ParagraphStyle("h1", fontName="Sans-Bold", fontSize=13, leading=17, spaceBefore=14, spaceAfter=8),
    "h2": ParagraphStyle("h2", fontName="Sans-Bold", fontSize=11, leading=15, spaceBefore=6, spaceAfter=4),
    "body": ParagraphStyle("body", fontName="Sans", fontSize=11, leading=16.5, alignment=TA_JUSTIFY, spaceAfter=6),
    "step": ParagraphStyle("step", fontName="Sans", fontSize=11, leading=16, leftIndent=46, firstLineIndent=-46,
                           spaceAfter=3),
    "bullet": ParagraphStyle("bullet", fontName="Sans", fontSize=11, leading=16, leftIndent=18, firstLineIndent=-12,
                             spaceAfter=5, alignment=TA_JUSTIFY),
    "cap": ParagraphStyle("cap", fontName="Sans-Italic", fontSize=9.5, leading=12, alignment=TA_CENTER,
                          spaceBefore=4, spaceAfter=14),
    "code": ParagraphStyle("code", fontName="Mono", fontSize=8, leading=10.2),
    "cell": ParagraphStyle("cell", fontName="Sans", fontSize=10, leading=13),
    "cellb": ParagraphStyle("cellb", fontName="Sans-Bold", fontSize=10, leading=13),
}

_fig = [0]


def para(t, st="body"):
    return Paragraph(t, S[st] if isinstance(st, str) else st)


def steps(items, label="Step"):
    return [para(f"<b>{label} {i}:</b>&nbsp;&nbsp;{t}", "step") for i, t in enumerate(items, 1)]


def bullets(items):
    return [para(f"•&nbsp;&nbsp;{t}", "bullet") for t in items]


def image(path, caption=None, width=CONTENT_W, max_h=190 * mm):
    im = PILImage.open(path)
    w, h = im.size
    sc = min(width / w, max_h / h)
    src = path
    if os.path.getsize(path) > 300_000:                  # large figures -> JPEG to keep the PDF small
        src = io.BytesIO()
        im.convert("RGB").save(src, "JPEG", quality=90)
        src.seek(0)
    items = [Image(src, w * sc, h * sc)]
    if caption:
        _fig[0] += 1
        items.append(para(f"Figure {_fig[0]}: {caption}", "cap"))
    return KeepTogether(items)


def table(rows, col_widths=None):
    data = [[Paragraph(str(c), S["cellb"] if r == 0 else S["cell"]) for c in row] for r, row in enumerate(rows)]
    t = Table(data, colWidths=col_widths, hAlign="CENTER", repeatRows=1)
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.6, colors.black),
                           ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                           ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                           ("LEFTPADDING", (0, 0), (-1, -1), 6)]))
    return t


def heading(num, title):
    return para(f"<u>{num}. {title.upper()}</u>", "h1")


def code_block(path):
    import textwrap
    max_chars = int((CONTENT_W - 18) / (0.6 * S["code"].fontSize))
    lines = []
    for line in open(path).read().rstrip().splitlines():
        if len(line) <= max_chars:
            lines.append(line or " ")
        else:
            ind = " " * (len(line) - len(line.lstrip()) + 4)
            lines += textwrap.wrap(line, max_chars, subsequent_indent=ind, drop_whitespace=False) or [" "]
    box = Table([[Preformatted(l, S["code"]) if l.strip() else Spacer(1, S["code"].leading)] for l in lines],
                colWidths=[CONTENT_W])
    box.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.6, colors.black),
                             ("TOPPADDING", (0, 0), (-1, -1), 0), ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
                             ("TOPPADDING", (0, 0), (-1, 0), 6), ("BOTTOMPADDING", (0, -1), (-1, -1), 6),
                             ("LEFTPADDING", (0, 0), (-1, -1), 8)]))
    return [para(f"<b>File name:</b> {os.path.basename(path)}"), box]


def terminal_png(text, cmd, path):
    """Render the program's real console output as a terminal-window screenshot."""
    font = ImageFont.truetype(f"{LIB}/LiberationMono-Regular.ttf", 16)
    lines = [f"$ {cmd}"] + text.rstrip().splitlines()
    cw, lh = font.getbbox("M")[2], 22
    W, H = cw * max(len(l) for l in lines) + 40, lh * len(lines) + 56
    im = PILImage.new("RGB", (W, H), "#0C0C0C")
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, W, 28], fill="#2D2D2D")
    d.text((12, 5), "Terminal", fill="#DDDDDD", font=font)
    for i, l in enumerate(lines):
        d.text((18, 40 + i * lh), l, fill="#FFFFFF" if i == 0 else "#CCCCCC", font=font)
    im.save(path)
    return path


def cover(unit_no, unit_title, task):
    cs = ParagraphStyle("c", parent=S["cell"], fontSize=12.5, leading=16)
    rows = [[Paragraph(f"<b>{k}</b>", cs), Paragraph("<b>:</b>", cs), Paragraph(v, cs)] for k, v in STUDENT]
    t = Table(rows, colWidths=[50 * mm, 8 * mm, 70 * mm], hAlign="CENTER")
    t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                           ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7)]))
    line = Table([[""]], colWidths=[CONTENT_W], style=[("LINEBELOW", (0, 0), (-1, -1), 1, colors.black)])
    return [Spacer(1, 25 * mm),
            para("COMPUTER VISION", "title"),
            para("Subject Code: 21CSE454T", "center"),
            Spacer(1, 4 * mm), line, Spacer(1, 14 * mm),
            para(f"<u>ASSIGNMENT - UNIT {unit_no}</u>", "subtitle"),
            Spacer(1, 3 * mm),
            para(unit_title, ParagraphStyle("ut", parent=S["center"], fontSize=14, leading=20)),
            Spacer(1, 14 * mm),
            para("<b>Task:</b>", "body"),
            para(task, "body"),
            Spacer(1, 18 * mm),
            para("<u>SUBMITTED BY</u>", "subtitle"),
            Spacer(1, 6 * mm), t,
            PageBreak()]


def build(out_pdf, unit_no, unit_title, task, story):
    _fig[0] = 0

    def footer(c, doc):
        if doc.page > 1:
            c.setFont("Sans", 9)
            c.drawCentredString(PAGE_W / 2, 12 * mm, str(doc.page))

    doc = SimpleDocTemplate(out_pdf, pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN,
                            topMargin=20 * mm, bottomMargin=20 * mm,
                            title=f"Computer Vision Assignment - Unit {unit_no}: {unit_title}",
                            author="Srevarshan (RA2311026050039)")
    doc.build(cover(unit_no, unit_title, task) + story, onFirstPage=footer, onLaterPages=footer)
    print("Wrote", out_pdf)
