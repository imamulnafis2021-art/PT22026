"""Central configuration for the AI Document OCR Translator."""
from dataclasses import dataclass
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# --- Translation model (downloaded automatically from the Hugging Face Hub) ---
NLLB_MODEL_NAME = "facebook/nllb-200-distilled-600M"
TARGET_NLLB_CODE = "eng_Latn"
MAX_INPUT_TOKENS = 256
NUM_BEAMS = 2
TRANSLATION_BATCH_SIZE = 16

# --- Input handling ---
ALLOWED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".pdf")
MAX_PDF_PAGES = 10
PDF_RENDER_DPI = 150
MAX_IMAGE_SIDE = 2800  # larger images are downscaled to save GPU memory

# --- OCR ---
MIN_CONFIDENCE = 0.2  # OCR boxes below this confidence are dropped
OCR_BATCH_SIZE = 8
AUTO_DETECT_LABEL = "Auto-detect"
ENGLISH_LABEL = "English"


@dataclass(frozen=True)
class Language:
    """Per-language settings: EasyOCR codes, NLLB code and how OCR boxes are joined."""
    easyocr_codes: tuple
    nllb_code: str
    joiner: str = " "  # CJK text is joined without spaces


LANGUAGES = {
    "Chinese (Simplified)": Language(("ch_sim", "en"), "zho_Hans", ""),
    "Chinese (Traditional)": Language(("ch_tra", "en"), "zho_Hant", ""),
    "Japanese": Language(("ja", "en"), "jpn_Jpan", ""),
    "Korean": Language(("ko", "en"), "kor_Hang"),
    "Bangla": Language(("bn", "en"), "ben_Beng"),
    "Hindi": Language(("hi", "en"), "hin_Deva"),
    "Arabic": Language(("ar", "en"), "arb_Arab"),
    "Russian": Language(("ru", "en"), "rus_Cyrl"),
    "French": Language(("fr", "en"), "fra_Latn"),
    "German": Language(("de", "en"), "deu_Latn"),
    "Spanish": Language(("es", "en"), "spa_Latn"),
}

# Languages probed in auto-detect mode. Every extra entry costs one more OCR pass,
# so add more (e.g. "Hindi", "Arabic", "Russian") only if you need them.
AUTO_CANDIDATES = ("Chinese (Simplified)", "Japanese", "Korean", "Bangla")

# One extra probe for Latin-script documents (EasyOCR's shared 'latin' model).
LATIN_PROBE_CODES = ("fr", "en")
LATIN_LANGUAGES = ("French", "German", "Spanish", ENGLISH_LABEL)
# --- User-friendliness ---
LOW_CONFIDENCE_THRESHOLD = 0.5   # warn the user when average OCR confidence is below this
MIN_DIRECT_TEXT_CHARS = 30       # a PDF page with at least this much real text skips OCR
PRELOAD_MODELS = True            # load all models when the app starts