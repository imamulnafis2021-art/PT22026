"""Turn uploaded files (images or PDFs) into a list of page images."""
from pathlib import Path
from typing import List

from PIL import Image

from config import ALLOWED_EXTENSIONS, MAX_PDF_PAGES, PDF_RENDER_DPI
from utils.image_processor import load_image, prepare_image


def pdf_to_images(path, dpi: int = PDF_RENDER_DPI, max_pages: int = MAX_PDF_PAGES) -> List[Image.Image]:
    """Render PDF pages to PIL images with PyMuPDF (no Poppler needed on Windows)."""
    import pymupdf  # PyMuPDF

    pages = []
    with pymupdf.open(str(path)) as doc:
        if doc.page_count == 0:
            raise ValueError("The PDF has no pages.")
        for index in range(min(doc.page_count, max_pages)):
            pix = doc[index].get_pixmap(dpi=dpi, alpha=False)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            pages.append(prepare_image(img))
    return pages


def load_pages(path, max_pages: int = MAX_PDF_PAGES) -> List[Image.Image]:
    """Load a JPG/PNG/PDF file as a list of page images."""
    path = Path(path)
    ext = path.suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Unsupported file type '{ext}'. Use JPG, PNG or PDF.")
    if ext == ".pdf":
        return pdf_to_images(path, max_pages=max_pages)
    return [load_image(path)]
def extract_pdf_text(path, max_pages: int = MAX_PDF_PAGES) -> List[str]:
    """Return the built-in text of each PDF page ('' for scanned pages that have no text layer)."""
    import pymupdf

    texts = []
    with pymupdf.open(str(path)) as doc:
        for index in range(min(doc.page_count, max_pages)):
            texts.append(doc[index].get_text("text").strip())
    return texts