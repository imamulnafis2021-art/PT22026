"""Language identification from OCR text (Unicode script analysis + lingua for Latin text)."""
import re
from functools import lru_cache
from typing import Optional

from config import ENGLISH_LABEL

_SCRIPTS = {
    "kana": re.compile(r"[\u3040-\u30ff\u31f0-\u31ff]"),
    "hangul": re.compile(r"[\uac00-\ud7af\u1100-\u11ff\u3130-\u318f]"),
    "bengali": re.compile(r"[\u0980-\u09ff]"),
    "devanagari": re.compile(r"[\u0900-\u097f]"),
    "arabic": re.compile(r"[\u0600-\u06ff\u0750-\u077f]"),
    "cyrillic": re.compile(r"[\u0400-\u04ff]"),
    "han": re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]"),
    "latin": re.compile(r"[A-Za-z\u00c0-\u00ff]"),
}

_SCRIPT_TO_LANGUAGE = {
    "hangul": "Korean",
    "bengali": "Bangla",
    "devanagari": "Hindi",
    "arabic": "Arabic",
    "cyrillic": "Russian",
    "han": "Chinese (Simplified)",
}


@lru_cache(maxsize=1)
def _latin_detector():
    """Build the lingua detector lazily (loading it takes a moment)."""
    from lingua import Language as L
    from lingua import LanguageDetectorBuilder

    detector = LanguageDetectorBuilder.from_languages(
        L.ENGLISH, L.FRENCH, L.GERMAN, L.SPANISH
    ).build()
    names = {L.ENGLISH: ENGLISH_LABEL, L.FRENCH: "French", L.GERMAN: "German", L.SPANISH: "Spanish"}
    return detector, names


def script_counts(text: str) -> dict:
    """Count characters of each supported Unicode script in the text."""
    return {name: len(pattern.findall(text)) for name, pattern in _SCRIPTS.items()}


def detect_language(text: str) -> Optional[str]:
    """Return a language name from config.LANGUAGES (or 'English'), or None if unknown."""
    if not text or not text.strip():
        return None
    counts = script_counts(text)

    # Japanese mixes kana with kanji (Han); any meaningful kana share means Japanese.
    cjk_total = counts["kana"] + counts["han"]
    if counts["kana"] > 0 and counts["kana"] >= 0.1 * cjk_total:
        return "Japanese"

    non_latin = {k: v for k, v in counts.items() if k not in ("latin", "kana") and v > 0}
    if non_latin:
        top = max(non_latin, key=non_latin.get)
        # Ignore a few stray symbols inside mostly Latin text.
        if counts[top] >= 0.25 * counts["latin"]:
            return _SCRIPT_TO_LANGUAGE[top]

    if counts["latin"] > 0:
        detector, names = _latin_detector()
        guess = detector.detect_language_of(text)
        return names.get(guess)
    return None
