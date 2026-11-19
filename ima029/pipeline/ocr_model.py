"""OCR stage: EasyOCR (CRAFT text detection + CRNN text recognition)."""
import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

from config import (
    AUTO_CANDIDATES,
    AUTO_DETECT_LABEL,
    LANGUAGES,
    LATIN_LANGUAGES,
    LATIN_PROBE_CODES,
    MIN_CONFIDENCE,
    OCR_BATCH_SIZE,
)
from utils.language_detector import detect_language

logger = logging.getLogger(__name__)

# One EasyOCR box: (4 corner points, text, confidence)
RawResult = Tuple[list, str, float]


@dataclass
class OcrResult:
    text: str
    language: str
    avg_confidence: float
    annotated: Image.Image
    num_boxes: int


def _joiner(language: str) -> str:
    return LANGUAGES[language].joiner if language in LANGUAGES else " "


def group_lines(results: List[RawResult], joiner: str = " ") -> str:
    """Sort OCR boxes top-to-bottom, group boxes on the same row, and join them into text."""
    items = []
    for bbox, text, _conf in results:
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        if text.strip():
            items.append({
                "x": min(xs),
                "yc": (min(ys) + max(ys)) / 2,
                "h": max(ys) - min(ys),
                "text": text.strip(),
            })
    if not items:
        return ""
    items.sort(key=lambda d: d["yc"])

    rows, current = [], [items[0]]
    for item in items[1:]:
        ref = current[-1]
        if abs(item["yc"] - ref["yc"]) <= 0.5 * max(item["h"], ref["h"]):
            current.append(item)
        else:
            rows.append(current)
            current = [item]
    rows.append(current)

    return "\n".join(joiner.join(i["text"] for i in sorted(row, key=lambda d: d["x"])) for row in rows)


def _score(results: List[RawResult]) -> float:
    """Confidence-weighted character count: higher means 'this reader understood the page'."""
    return sum(conf * len(text.strip()) for _b, text, conf in results)


def draw_boxes(image: Image.Image, results: List[RawResult]) -> Image.Image:
    """Return a copy of the image with the detected text boxes drawn on it."""
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    width = max(2, annotated.width // 500)
    for bbox, _text, _conf in results:
        points = [(float(x), float(y)) for x, y in bbox]
        draw.line(points + [points[0]], fill=(255, 0, 0), width=width)
    return annotated


class OcrEngine:
    """Lazy-loading wrapper around EasyOCR readers (one reader per language group)."""

    def __init__(self, use_gpu: Optional[bool] = None):
        if use_gpu is None:
            import torch
            use_gpu = torch.cuda.is_available()
        self.use_gpu = use_gpu
        self._readers = {}

    def _reader(self, codes: tuple):
        if codes not in self._readers:
            import easyocr
            logger.info("Loading EasyOCR reader for %s (gpu=%s)", codes, self.use_gpu)
            self._readers[codes] = easyocr.Reader(list(codes), gpu=self.use_gpu, verbose=False)
        return self._readers[codes]

    def _read(self, image: Image.Image, codes: tuple) -> List[RawResult]:
        raw = self._reader(codes).readtext(np.array(image), batch_size=OCR_BATCH_SIZE)
        return [(b, t, float(c)) for b, t, c in raw if float(c) >= MIN_CONFIDENCE and t.strip()]

    def _auto_detect(self, image: Image.Image) -> Tuple[str, List[RawResult]]:
        """Try each candidate reader; keep the most confident one whose text matches its own script."""
        candidates = [(name, LANGUAGES[name].easyocr_codes) for name in AUTO_CANDIDATES]
        candidates.append(("Latin", LATIN_PROBE_CODES))

        best_consistent = None
        best_any = None
        for name, codes in candidates:
            results = self._read(image, codes)
            score = _score(results)
            if score <= 0:
                continue
            text = group_lines(results, _joiner(name))
            detected = detect_language(text)
            if name == "Latin":
                resolved, consistent = detected, detected in LATIN_LANGUAGES
            else:
                resolved, consistent = name, detected == name
            if resolved is None:
                continue
            logger.info("Auto-detect probe %s: score=%.1f detected=%s consistent=%s",
                        name, score, detected, consistent)
            entry = (score, resolved, results)
            if best_any is None or score > best_any[0]:
                best_any = entry
            if consistent and (best_consistent is None or score > best_consistent[0]):
                best_consistent = entry

        chosen = best_consistent or best_any
        if chosen is None:
            raise ValueError("No text detected. Try a clearer image or choose the language manually.")
        return chosen[1], chosen[2]

    def extract(self, image: Image.Image, language_choice: str = AUTO_DETECT_LABEL) -> OcrResult:
        """Run OCR on one page image. Returns text, language and an annotated image."""
        if language_choice == AUTO_DETECT_LABEL:
            try:
                language, results = self._auto_detect(image)
            except ValueError:
                return OcrResult("", "Unknown", 0.0, image, 0)
        elif language_choice in LANGUAGES:
            language = language_choice
            results = self._read(image, LANGUAGES[language].easyocr_codes)
        else:
            raise ValueError(f"Unsupported language: {language_choice}")

        text = group_lines(results, _joiner(language))
        avg_conf = float(np.mean([c for _b, _t, c in results])) if results else 0.0
        return OcrResult(text, language, avg_conf, draw_boxes(image, results), len(results))
