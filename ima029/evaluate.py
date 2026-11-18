"""Evaluate the OCR + translation pipeline on tests/eval_data.

Files per sample (same base name):
    zh_01.jpg     image (or .png / .pdf)
    zh_01.txt     exact original text (UTF-8)
    zh_01.en.txt  English reference translation (UTF-8)

Run from the ima029 folder with the venv active:
    python evaluate.py              # language chosen from the file prefix
    python evaluate.py --auto       # use Auto-detect instead

Output: docs/eval_results.csv (one row per sample) and a summary in the terminal.
Requires: pip install sacrebleu
"""
import os

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import argparse
import csv
import re
import time
from collections import defaultdict
from pathlib import Path

import sacrebleu

from config import AUTO_DETECT_LABEL, LANGUAGES
from pipeline.ocr_model import OcrEngine
from pipeline.translation_model import Translator
from utils.pdf_processor import load_pages

BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "tests" / "eval_data"
OUT_CSV = BASE / "docs" / "eval_results.csv"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".pdf"}

# file prefix -> keyword searched for in the LANGUAGES names of config.py
PREFIX_KEYWORDS = {
    "zh": ["chinese"],
    "ja": ["japanese"],
    "ko": ["korean"],
    "bn": ["bangla", "bengali"],
}


def language_for(prefix):
    for name in LANGUAGES:
        if any(k in name.lower() for k in PREFIX_KEYWORDS.get(prefix, [])):
            return name
    return None


def read_text(path):
    return Path(path).read_text(encoding="utf-8-sig").strip()


def edit_distance(a, b):
    """Levenshtein distance between two sequences."""
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1,
                               previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def normalize(text):
    return re.sub(r"\s+", "", text)  # ignore spaces and line breaks


def cer(reference, hypothesis):
    ref, hyp = normalize(reference), normalize(hypothesis)
    return edit_distance(ref, hyp) / max(len(ref), 1)


def wer(reference, hypothesis):
    ref, hyp = reference.split(), hypothesis.split()
    return edit_distance(ref, hyp) / max(len(ref), 1)


def peak_vram_mb():
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1024 ** 2
    except Exception:
        pass
    return 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--auto", action="store_true", help="use Auto-detect instead of the file prefix")
    args = parser.parse_args()

    images = sorted(p for p in DATA_DIR.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not images:
        raise SystemExit(f"No images found in {DATA_DIR}")

    ocr = OcrEngine()
    translator = Translator()
    rows = []

    for img_path in images:
        stem = img_path.stem
        gt_path = DATA_DIR / f"{stem}.txt"
        ref_path = DATA_DIR / f"{stem}.en.txt"
        if not gt_path.exists() or not ref_path.exists():
            print(f"SKIP {img_path.name}: missing {gt_path.name} or {ref_path.name}")
            continue

        prefix = stem.split("_")[0].lower()
        language = AUTO_DETECT_LABEL if args.auto else language_for(prefix)
        if language is None:
            print(f"SKIP {img_path.name}: unknown language prefix '{prefix}'")
            continue

        truth = read_text(gt_path)
        reference = read_text(ref_path)
        page = load_pages(img_path)[0]

        t0 = time.time()
        result = ocr.extract(page, language)
        t_ocr = time.time() - t0

        detected = result.language
        t0 = time.time()
        # End-to-end: translate the OCR output
        e2e = translator.translate(result.text, detected) if result.text.strip() else ""
        t_tr = time.time() - t0
        # Translation only: translate the perfect ground-truth text
        clean = translator.translate(truth, detected)

        row = {
            "file": img_path.name,
            "lang": prefix,
            "requested": language,
            "detected": detected,
            "lang_correct": int(language_for(prefix) == detected),
            "ocr_conf": round(result.avg_confidence, 3),
            "cer": round(cer(truth, result.text), 4),
            "wer": round(wer(truth, result.text), 4),
            "bleu_e2e": round(sacrebleu.sentence_bleu(e2e, [reference]).score, 2),
            "chrf_e2e": round(sacrebleu.sentence_chrf(e2e, [reference]).score, 2),
            "bleu_clean": round(sacrebleu.sentence_bleu(clean, [reference]).score, 2),
            "chrf_clean": round(sacrebleu.sentence_chrf(clean, [reference]).score, 2),
            "ocr_s": round(t_ocr, 2),
            "translate_s": round(t_tr, 2),
            "peak_vram_mb": round(peak_vram_mb()),
            "ocr_text": result.text.replace("\n", " "),
            "translation": e2e.replace("\n", " "),
        }
        rows.append(row)
        print(f"{img_path.name}: CER={row['cer']:.3f} chrF={row['chrf_e2e']:.1f} "
              f"BLEU={row['bleu_e2e']:.1f} ({t_ocr:.1f}s OCR, {t_tr:.1f}s MT)")

    if not rows:
        raise SystemExit("Nothing was evaluated.")

    OUT_CSV.parent.mkdir(exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Summary per language and overall
    groups = defaultdict(list)
    for r in rows:
        groups[r["lang"]].append(r)
        groups["ALL"].append(r)
    keys = ["cer", "wer", "bleu_e2e", "chrf_e2e", "bleu_clean", "chrf_clean", "ocr_s", "translate_s"]
    print("\n=== Summary (means) ===")
    print("lang  n  " + "  ".join(f"{k:>10}" for k in keys) + "  lang_acc")
    for lang, items in groups.items():
        means = [sum(i[k] for i in items) / len(items) for k in keys]
        acc = sum(i["lang_correct"] for i in items) / len(items)
        print(f"{lang:<5}{len(items):>2}  " + "  ".join(f"{m:>10.3f}" for m in means) + f"  {acc:.0%}")
    print(f"\nSaved: {OUT_CSV}")


if __name__ == "__main__":
    main()