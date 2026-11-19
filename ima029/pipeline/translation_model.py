"""Translation : NLLB-200 (distilled 600M) encoder-decoder transformer, any language -> English."""
import logging
import re
from typing import List, Optional

from config import (
    ENGLISH_LABEL,
    LANGUAGES,
    MAX_INPUT_TOKENS,
    NLLB_MODEL_NAME,
    NUM_BEAMS,
    TARGET_NLLB_CODE,
    TRANSLATION_BATCH_SIZE,
)

logger = logging.getLogger(__name__)

# Split after sentence-ending punctuation (CJK, Bangla danda, ! ?) or after ". " followed by space.
_SENTENCE_SPLIT = re.compile(r"(?<=[。！？!?।])\s*|(?<=\.)\s+")


def split_sentences(text: str) -> List[str]:
    """Split one line of text into sentence-sized segments (NLLB works best on short inputs)."""
    return [s for s in _SENTENCE_SPLIT.split(text) if s and s.strip()]


class Translator:
    """Lazy-loading NLLB translator (fp16 on GPU, about 1.5 GB of VRAM)."""

    def __init__(self, model_name: str = NLLB_MODEL_NAME, device: Optional[str] = None):
        self.model_name = model_name
        self.device = device
        self._tokenizer = None
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Loading %s on %s", self.model_name, self.device)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        if self.device == "cuda":
            model = model.half()
        self._model = model.to(self.device).eval()

    def _translate_batch(self, segments: List[str]) -> List[str]:
        import torch

        inputs = self._tokenizer(
            segments, return_tensors="pt", padding=True, truncation=True, max_length=MAX_INPUT_TOKENS
        ).to(self.device)
        target_id = self._tokenizer.convert_tokens_to_ids(TARGET_NLLB_CODE)
        with torch.inference_mode():
            output = self._model.generate(
                **inputs,
                forced_bos_token_id=target_id,
                max_new_tokens=MAX_INPUT_TOKENS,
                num_beams=NUM_BEAMS,
            )
        return self._tokenizer.batch_decode(output, skip_special_tokens=True)

    def translate(self, text: str, language: str) -> str:
        """Translate text from `language` (a key of config.LANGUAGES) into English, keeping line breaks."""
        if not text.strip():
            return ""
        if language == ENGLISH_LABEL:
            return text
        if language not in LANGUAGES:
            raise ValueError(f"Unsupported source language: {language}")

        self._load()
        self._tokenizer.src_lang = LANGUAGES[language].nllb_code

        lines = text.split("\n")
        segments, owners = [], []
        for line_index, line in enumerate(lines):
            for sentence in split_sentences(line):
                segments.append(sentence.strip())
                owners.append(line_index)

        translated = []
        for start in range(0, len(segments), TRANSLATION_BATCH_SIZE):
            translated.extend(self._translate_batch(segments[start:start + TRANSLATION_BATCH_SIZE]))

        out_lines = [""] * len(lines)
        for line_index, piece in zip(owners, translated):
            out_lines[line_index] = (out_lines[line_index] + " " + piece).strip()
        return "\n".join(out_lines)
