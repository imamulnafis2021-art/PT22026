"""AI Document OCR Translator - Gradio app.

Upload a JPG/PNG/PDF in Chinese, Japanese, Korean, Bangla (and more), extract the text with
EasyOCR (or directly from digital PDFs) and translate it into English with NLLB-200.
"""
import os

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")  # avoids the failing Xet downloader

import logging
import re
import time
from datetime import datetime
from pathlib import Path

import gradio as gr

from config import (
    ALLOWED_EXTENSIONS,
    AUTO_CANDIDATES,
    AUTO_DETECT_LABEL,
    BASE_DIR,
    ENGLISH_LABEL,
    LANGUAGES,
    LATIN_PROBE_CODES,
    LOW_CONFIDENCE_THRESHOLD,
    MIN_DIRECT_TEXT_CHARS,
    OUTPUT_DIR,
    PRELOAD_MODELS,
)
from pipeline.ocr_model import OcrEngine
from pipeline.translation_model import Translator
from utils.exporter import save_pdf, save_txt
from utils.language_detector import detect_language
from utils.pdf_processor import extract_pdf_text, load_pages

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("app")

ocr_engine = OcrEngine()
translator = Translator()

LANGUAGE_CHOICES = [AUTO_DETECT_LABEL] + list(LANGUAGES)
SAMPLES_DIR = BASE_DIR / "tests" / "samples"


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def _free_gpu_memory():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # never let cleanup break the app
        pass


def _page_labels(total):
    return [f"Page {i}" for i in range(1, total + 1)]


def _page_index(label, total):
    try:
        index = int(str(label).split()[-1]) - 1
    except (ValueError, IndexError):
        index = 0
    return min(max(index, 0), max(total - 1, 0))


def _combine(parts):
    parts = [p if p.strip() else "(no text detected)" for p in parts]
    if len(parts) == 1:
        return parts[0]
    return "\n\n".join(f"--- Page {i} ---\n{p}" for i, p in enumerate(parts, start=1))


def _clean_stem(name):
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", Path(name).stem).strip("_")
    return stem[:40] or "document"


def _build_downloads(state):
    """Write TXT and PDF files from the current (possibly edited) state."""
    pages = state["pages"]
    original = _combine([p["original"] for p in pages])
    translation = _combine([p["translation"] for p in pages])
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"{state['stem']}_translation_{stamp}"
    txt_path = save_txt(OUTPUT_DIR / f"{base}.txt", original, translation, state["language"])
    pdf_path = save_pdf(OUTPUT_DIR / f"{base}.pdf", translation, state["language"])
    return [str(txt_path), str(pdf_path)]


def _read_page(image, direct_text, language):
    """Read one page: use the PDF's own text layer when it has one, otherwise run OCR."""
    if len(direct_text) >= MIN_DIRECT_TEXT_CHARS:
        page_language = language or detect_language(direct_text)
        if page_language == ENGLISH_LABEL or page_language in LANGUAGES:
            return {"original": direct_text, "language": page_language,
                    "confidence": 1.0, "image": image, "source": "PDF text"}
    ocr_choice = language if language in LANGUAGES else AUTO_DETECT_LABEL
    result = ocr_engine.extract(image, ocr_choice)
    return {"original": result.text, "language": result.language,
            "confidence": result.avg_confidence, "image": result.annotated, "source": "OCR"}


# ----------------------------------------------------------------------------
# Main pipeline
# ----------------------------------------------------------------------------
def process(file_path, language_choice, progress=gr.Progress()):
    """File -> pages -> (PDF text or OCR) -> translation -> exports."""
    if not file_path:
        raise gr.Error("Please upload a JPG, PNG or PDF file first.")
    path = Path(file_path)
    if path.suffix.lower() not in ALLOWED_EXTENSIONS:
        raise gr.Error("Unsupported file type. Please upload a JPG, PNG or PDF.")
    is_pdf = path.suffix.lower() == ".pdf"

    started = time.time()
    try:
        progress(0.05, desc="Loading file...")
        images = load_pages(path)
        direct_texts = extract_pdf_text(path, max_pages=len(images)) if is_pdf else []
    except Exception as exc:
        logger.exception("Could not load file")
        raise gr.Error(f"Could not read the file ({exc}). It may be corrupted or password-protected.")

    total = len(images)
    language = None if language_choice == AUTO_DETECT_LABEL else language_choice
    entries = []

    try:
        for number, image in enumerate(images, start=1):
            progress(0.1 + 0.8 * (number - 1) / total, desc=f"Reading page {number}/{total}...")
            direct = direct_texts[number - 1] if number <= len(direct_texts) else ""
            entry = _read_page(image, direct, language)

            if entry["original"].strip():
                language = language or entry["language"]  # reuse the language found first
                progress(0.1 + 0.8 * (number - 0.4) / total, desc=f"Translating page {number}/{total}...")
                entry["translation"] = translator.translate(entry["original"], entry["language"])
            else:
                entry["translation"] = ""
            entries.append(entry)
            _free_gpu_memory()
    except gr.Error:
        raise
    except Exception as exc:
        logger.exception("Pipeline failed")
        raise gr.Error(f"Processing failed: {exc}")

    if not any(e["original"].strip() for e in entries):
        raise gr.Error("No text was detected. Try a clearer or higher-resolution image, "
                       "or choose the language manually.")

    state = {"language": language, "pages": entries, "stem": _clean_stem(path.name)}

    # Build the summary line and warn about low-quality pages.
    ocr_confs = [e["confidence"] for e in entries if e["source"] == "OCR" and e["original"].strip()]
    low_pages = [i for i, e in enumerate(entries, start=1)
                 if e["source"] == "OCR" and e["original"].strip() and e["confidence"] < LOW_CONFIDENCE_THRESHOLD]
    pdf_text_pages = sum(1 for e in entries if e["source"] == "PDF text")

    parts = [f"**Detected language:** {language}", f"**Pages:** {total}"]
    if ocr_confs:
        parts.append(f"**Avg. OCR confidence:** {sum(ocr_confs) / len(ocr_confs):.2f}")
    if pdf_text_pages:
        parts.append(f"**Read directly from PDF text:** {pdf_text_pages} page(s)")
    parts.append(f"**Time:** {time.time() - started:.1f}s")
    info = "  |  ".join(parts)

    if low_pages:
        pages_text = ", ".join(str(p) for p in low_pages)
        message = (f"Low OCR confidence on page(s) {pages_text}. The image may be blurry, tilted or low "
                   "resolution. Try a clearer scan, choose the language manually, or fix the text below "
                   "and press 'Re-translate edited text'.")
        info += f"\n\n**Warning:** {message}"
        gr.Warning(message)

    labels = _page_labels(total)
    first = entries[0]
    return (
        state,
        gr.update(choices=labels, value=labels[0], visible=total > 1),
        first["image"],
        info,
        first["original"],
        first["translation"],
        _build_downloads(state),
    )


# ----------------------------------------------------------------------------
# Interaction handlers
# ----------------------------------------------------------------------------
def show_page(page_label, state):
    """Show the image, original text and translation of the selected page."""
    if not state:
        return None, "", ""
    entry = state["pages"][_page_index(page_label, len(state["pages"]))]
    return entry["image"], entry["original"], entry["translation"]


def save_edit(text, page_label, state):
    """Remember edits to the original text so they survive page switches."""
    if state:
        state["pages"][_page_index(page_label, len(state["pages"]))]["original"] = text
    return state


def retranslate(text, page_label, state, progress=gr.Progress()):
    """Translate the (edited) original text of the current page again, without re-running OCR."""
    if not state:
        raise gr.Error("Please run 'Extract and Translate' first.")
    if not text or not text.strip():
        raise gr.Error("The original text box is empty.")

    entry = state["pages"][_page_index(page_label, len(state["pages"]))]
    language = entry["language"]
    if language not in LANGUAGES and language != ENGLISH_LABEL:
        language = state["language"]
    if language not in LANGUAGES and language != ENGLISH_LABEL:
        raise gr.Error("The source language is unknown. Run again and choose the language manually.")

    progress(0.3, desc="Translating...")
    try:
        translation = translator.translate(text, language)
    except Exception as exc:
        logger.exception("Re-translation failed")
        raise gr.Error(f"Translation failed: {exc}")

    entry["original"] = text
    entry["translation"] = translation
    _free_gpu_memory()
    return state, translation, _build_downloads(state)


def clear_all():
    return (None, AUTO_DETECT_LABEL, None, gr.update(choices=[], value=None, visible=False),
            None, "", "", "", None)


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
def _sample_examples():
    if not SAMPLES_DIR.exists():
        return []
    return [[str(p), AUTO_DETECT_LABEL] for p in sorted(SAMPLES_DIR.iterdir())
            if p.suffix.lower() in ALLOWED_EXTENSIONS]


def build_ui() -> gr.Blocks:
    custom_css = """
    textarea {
        font-family: 'Times New Roman', Times, serif !important;
        font-size: 16px !important;
    }
    """

    with gr.Blocks(
        title="AI Document OCR Translator",
        theme=gr.themes.Soft(),
        css=custom_css
    ) as demo:

        state = gr.State(None)

        gr.Markdown(
            "# AI Document OCR Translator by IMAMUL-20243120029\n"
            "Upload a **JPG, PNG or PDF** (Chinese, Japanese, Korean, Bangla and more). "
            "The app extracts the text with **EasyOCR** and translates it into English with **NLLB-200**. "
            "OCR mistakes? Edit the original text and press **Re-translate edited text**."
        )

        with gr.Row():
            with gr.Column(scale=1):
                file_input = gr.File(
                    label="Upload document",
                    file_types=list(ALLOWED_EXTENSIONS),
                    type="filepath"
                )

                language_input = gr.Dropdown(
                    LANGUAGE_CHOICES,
                    value=AUTO_DETECT_LABEL,
                    label="Source language"
                )

                run_button = gr.Button(
                    "Extract and Translate",
                    variant="primary"
                )

                clear_button = gr.Button("Clear")

            with gr.Column(scale=2):
                page_input = gr.Dropdown(
                    choices=[],
                    label="Page",
                    visible=False,
                    interactive=True
                )

                image_output = gr.Image(
                    label="Page preview (detected text boxes)",
                    type="pil",
                    height=360,
                    interactive=False
                )

        info_output = gr.Markdown()

        with gr.Row():
            original_output = gr.Textbox(
                label="Original text (editable)",
                lines=16,
                show_copy_button=True,
                interactive=True
            )

            translated_output = gr.Textbox(
                label="English translation",
                lines=16,
                show_copy_button=True,
                interactive=False
            )

        retranslate_button = gr.Button("Re-translate edited text")

        downloads = gr.File(
            label="Download results (TXT and PDF)",
            file_count="multiple"
        )

        examples = _sample_examples()

        if examples:
            gr.Examples(
                examples=examples,
                inputs=[file_input, language_input],
                label="Try an example (click one, then press Extract and Translate)"
            )

        run_button.click(
            process,
            inputs=[file_input, language_input],
            outputs=[
                state,
                page_input,
                image_output,
                info_output,
                original_output,
                translated_output,
                downloads
            ],
        )

        page_input.input(
            show_page,
            [page_input, state],
            [image_output, original_output, translated_output]
        )

        original_output.input(
            save_edit,
            [original_output, page_input, state],
            state,
            trigger_mode="always_last",
            show_progress="hidden"
        )

        retranslate_button.click(
            retranslate,
            [original_output, page_input, state],
            [state, translated_output, downloads]
        )

        clear_button.click(
            clear_all,
            inputs=None,
            outputs=[
                file_input,
                language_input,
                state,
                page_input,
                image_output,
                info_output,
                original_output,
                translated_output,
                downloads
            ],
        )

    return demo


def preload_models():
    """Load the translation and OCR models once at start-up so the first request is fast."""
    try:
        logger.info("Loading models (the first start can take a while)...")
        translator._load()
        for name in AUTO_CANDIDATES:
            ocr_engine._reader(LANGUAGES[name].easyocr_codes)
        ocr_engine._reader(LATIN_PROBE_CODES)
        logger.info("Models ready.")
    except Exception:
        logger.exception("Preloading failed; models will load on first use instead.")


if __name__ == "__main__":
    if PRELOAD_MODELS:
        preload_models()
    build_ui().queue().launch()