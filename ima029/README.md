# AI Document OCR Translator

Upload a scanned document or photo (**JPG, PNG, PDF**) written in Chinese, Japanese, Korean, Bangla
(and more), extract the text with OCR and get an **English translation** side by side with the original.

Built for the Professional Training mid-term assignment (Gradio + Hugging Face).

## Features
- Upload JPG, PNG and multi-page PDF files
- Automatic language detection (or choose the language manually)
- OCR with detected text boxes drawn on the page
- Translation into English
- Side-by-side original and translated text with copy buttons
- Download results as TXT (original + translation) or PDF (English translation)

## Models
| Stage | Model | Notes |
|---|---|---|
| OCR | [EasyOCR](https://github.com/JaidedAI/EasyOCR) (CRAFT detector + CRNN recognizer) | 80+ languages, downloaded on first use |
| Translation | [`facebook/nllb-200-distilled-600M`](https://huggingface.co/facebook/nllb-200-distilled-600M) | 200 languages, about 1.5 GB VRAM in fp16 |
| Language ID | Unicode script analysis + [lingua](https://github.com/pemistahl/lingua-py) | script rules for CJK/Bangla, lingua for Latin text |

Model weights are **not** stored in this repository. They download automatically the first time you
run the app (EasyOCR to `~/.EasyOCR`, NLLB to the Hugging Face cache), so the first run needs internet.

## Requirements
- Python 3.11.9
- NVIDIA GPU recommended (tested on RTX 4060, 8 GB). CPU also works but is slower.

## Installation (Windows PowerShell)
```powershell
cd <this folder>
python -m venv .venv
.venv\Scripts\activate

# 1) PyTorch with CUDA (choose the command for your system at https://pytorch.org/get-started/locally/)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 2) The rest
pip install -r requirements.txt

# 3) Check the GPU
python -c "import torch; print(torch.cuda.is_available())"
```

## Run
```powershell
python app.py
```
Open the local URL printed in the terminal (usually http://127.0.0.1:7860).

## Usage
1. Upload a JPG, PNG or PDF.
2. Keep **Auto-detect** or pick the source language.
3. Click **Extract and Translate**.
4. Read, copy or download the results.

## How auto-detect works
The text language is unknown before OCR, so the app runs a few EasyOCR readers
(Chinese, Japanese, Korean, Bangla and a Latin-script reader) on the first page. It keeps the reader with the
highest confidence-weighted text whose output matches its own script (kana means Japanese, Hangul means Korean, and so on).
For Latin text, lingua picks French, German, Spanish or English. Add languages to `AUTO_CANDIDATES` in `config.py` if needed;
each extra language adds one OCR pass.

## Project structure
```
app.py                    Gradio interface and pipeline orchestration
config.py                 Model names, language table and limits
pipeline/ocr_model.py     EasyOCR wrapper, auto-detection, text-box drawing
pipeline/translation_model.py   NLLB translator
utils/                    PDF/image loading, language detection, TXT/PDF export
tests/test_pipeline.py    Unit tests
outputs/                  Generated downloads (git-ignored)
docs/screenshots/         Screenshots for the report
```

## Tests
```powershell
python -m pytest -v
set RUN_SLOW=1   # optional: also runs the NLLB end-to-end test
```

## Limitations
- Handwriting, low-resolution scans, vertical Japanese text and complex layouts (tables, columns) reduce accuracy.
- Translation quality is lower for noisy OCR text and low-resource languages.
- The PDF export contains the English translation only, because standard PDF fonts cannot draw CJK/Bangla text; use the TXT file for the original.
- PDFs are limited to the first 10 pages (`MAX_PDF_PAGES` in `config.py`).

## Acknowledgements
EasyOCR (Jaided AI), NLLB-200 (Meta AI), Hugging Face Transformers, Gradio, PyMuPDF, ReportLab, lingua-py.
