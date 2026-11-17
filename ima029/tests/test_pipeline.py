"""Unit tests. Run from the project folder with:  python -m pytest -v

Set RUN_SLOW=1 to also run the test that downloads and runs the NLLB model.
"""
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image  # noqa: E402

from pipeline.ocr_model import group_lines  # noqa: E402
from pipeline.translation_model import split_sentences  # noqa: E402
from utils.exporter import save_pdf, save_txt  # noqa: E402
from utils.image_processor import limit_size  # noqa: E402
from utils.language_detector import detect_language  # noqa: E402
from utils.pdf_processor import load_pages  # noqa: E402


@pytest.mark.parametrize("text,expected", [
    ("你好，世界。今天天气很好。", "Chinese (Simplified)"),
    ("こんにちは、世界。今日はいい天気です。", "Japanese"),
    ("안녕하세요. 오늘 날씨가 좋습니다.", "Korean"),
    ("আমি বাংলায় গান গাই। আজ আবহাওয়া ভালো।", "Bangla"),
    ("Bonjour tout le monde, nous allons au marché ce matin pour acheter du pain.", "French"),
])
def test_detect_language(text, expected):
    assert detect_language(text) == expected


def test_detect_language_empty():
    assert detect_language("   ") is None


def test_split_sentences():
    assert split_sentences("你好。再见！") == ["你好。", "再见！"]
    assert split_sentences("Version 3.5 is out. Please update.") == ["Version 3.5 is out.", "Please update."]


def test_group_lines_orders_rows_and_columns():
    box = lambda x, y: [[x, y], [x + 40, y], [x + 40, y + 20], [x, y + 20]]  # noqa: E731
    results = [(box(100, 10), "world", 0.9), (box(0, 10), "hello", 0.9), (box(0, 60), "second line", 0.9)]
    assert group_lines(results) == "hello world\nsecond line"


def test_limit_size():
    img = Image.new("RGB", (6000, 3000))
    small = limit_size(img, 2800)
    assert max(small.size) == 2800 and small.size[0] / small.size[1] == pytest.approx(2.0, rel=0.01)


def test_load_pages_pdf_and_image(tmp_path):
    import pymupdf

    pdf_path = tmp_path / "sample.pdf"
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "Hello")
    doc.new_page().insert_text((72, 72), "World")
    doc.save(str(pdf_path))
    doc.close()
    assert len(load_pages(pdf_path)) == 2

    png_path = tmp_path / "pic.png"
    Image.new("RGB", (100, 50), "white").save(png_path)
    assert len(load_pages(png_path)) == 1


def test_load_pages_rejects_unknown_extension(tmp_path):
    bad = tmp_path / "notes.docx"
    bad.write_text("x")
    with pytest.raises(ValueError):
        load_pages(bad)


def test_exporters(tmp_path):
    txt = save_txt(tmp_path / "out.txt", "你好", "Hello", "Chinese (Simplified)")
    assert "你好" in txt.read_text(encoding="utf-8") and "Hello" in txt.read_text(encoding="utf-8")
    pdf = save_pdf(tmp_path / "out.pdf", "Hello <world> & “quotes”\n\nSecond paragraph", "Chinese (Simplified)")
    assert pdf.read_bytes().startswith(b"%PDF")


@pytest.mark.skipif(os.environ.get("RUN_SLOW") != "1", reason="set RUN_SLOW=1 to download and run NLLB")
def test_translation_end_to_end():
    from pipeline.translation_model import Translator

    result = Translator().translate("你好，世界。", "Chinese (Simplified)")
    assert "hello" in result.lower()
