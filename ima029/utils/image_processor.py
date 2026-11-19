"""Image loading and light pre-processing."""
from pathlib import Path

from PIL import Image, ImageOps

from config import MAX_IMAGE_SIDE


def limit_size(image: Image.Image, max_side: int = MAX_IMAGE_SIDE) -> Image.Image:
    """Downscale very large images (keeps aspect ratio) to limit memory use."""
    longest = max(image.size)
    if longest <= max_side:
        return image
    scale = max_side / longest
    new_size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
    return image.resize(new_size, Image.LANCZOS)


def prepare_image(image: Image.Image) -> Image.Image:
    """Fix orientation, convert to RGB and limit the size."""
    image = ImageOps.exif_transpose(image).convert("RGB")
    return limit_size(image)


def load_image(path) -> Image.Image:
    """Open an image file from disk and prepare it for OCR."""
    with Image.open(Path(path)) as img:
        img.load()
        return prepare_image(img)
