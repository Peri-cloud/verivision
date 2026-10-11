"""Pure image-preprocessing helpers shared by validators and detectors."""
from __future__ import annotations

from PIL import Image, ImageOps


def exif_transpose(img: Image.Image) -> Image.Image:
    """Apply the EXIF orientation tag (if any) so the image displays upright."""
    try:
        return ImageOps.exif_transpose(img)
    except Exception:
        # EXIF metadata can be malformed; the pixel data is still usable.
        return img


def ensure_rgb(img: Image.Image) -> Image.Image:
    """Convert any Pillow mode (P, RGBA, LA, CMYK, ...) to 3-channel RGB.

    Images with alpha are composited onto a plain white background so that
    transparent areas do not turn black after conversion.
    """
    if img.mode == "RGB":
        return img
    if img.mode in ("RGBA", "LA", "PA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        background.alpha_composite(rgba)
        return background.convert("RGB")
    return img.convert("RGB")


def image_stats(img: Image.Image) -> dict:
    """Basic dimension/mode information for the UI preview panel."""
    width, height = img.size
    return {
        "width": width,
        "height": height,
        "mode": img.mode,
        "aspect_ratio": round(width / height, 3) if height else None,
    }
