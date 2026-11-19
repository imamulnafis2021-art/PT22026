"""Export results as TXT or PDF."""
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer


def save_txt(path, original: str, translation: str, language: str) -> Path:
    """Save original text and English translation in one UTF-8 text file."""
    path = Path(path)
    content = (
        f"Detected language: {language}\n\n"
        f"===== ORIGINAL TEXT =====\n{original}\n\n"
        f"===== ENGLISH TRANSLATION =====\n{translation}\n"
    )
    path.write_text(content, encoding="utf-8")
    return path


def _pdf_safe(text: str) -> str:
    """Make text safe for ReportLab's built-in Helvetica font (Latin/WinAnsi only)."""
    text = text.encode("cp1252", "replace").decode("cp1252")
    return escape(text).replace("\n", "<br/>")


def save_pdf(path, translation: str, language: str) -> Path:
    """Save the English translation as a PDF.

    The built-in PDF fonts cannot draw Chinese/Japanese/Korean/Bangla (and ReportLab
    does not shape complex scripts), so the PDF contains the English text only.
    The original text is available in the TXT export.
    """
    path = Path(path)
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(str(path), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                            topMargin=2 * cm, bottomMargin=2 * cm, title="English translation")
    story = [
        Paragraph("English Translation", styles["Title"]),
        Paragraph(f"Source language: {_pdf_safe(language)}", styles["Italic"]),
        Spacer(1, 0.5 * cm),
    ]
    for block in translation.split("\n\n"):
        if block.strip():
            story.append(Paragraph(_pdf_safe(block.strip()), styles["BodyText"]))
            story.append(Spacer(1, 0.2 * cm))
    doc.build(story)
    return path
