"""Image loading and validation.

All uploads are validated from their raw BYTES (never by file extension) so
that a mislabelled file cannot sneak through. Each failure mode has a distinct
exception with a user-friendly message that the UI can show directly.
"""
from __future__ import annotations

import io

import config
from PIL import Image, UnidentifiedImageError
from PIL.Image import DecompressionBombError

from .preprocessing import ensure_rgb, exif_transpose


class ImageValidationError(ValueError):
    """Base class for all upload-validation failures."""

    def __init__(self, user_message: str, detail: str = ""):
        super().__init__(user_message)
        self.user_message = user_message
        self.detail = detail


class EmptyFileError(ImageValidationError):
    pass


class UnsupportedFormatError(ImageValidationError):
    pass


class CorruptImageError(ImageValidationError):
    pass


class FileTooLargeError(ImageValidationError):
    pass


class ImageDimensionsTooLargeError(ImageValidationError):
    pass


class ImageDimensionsTooSmallError(ImageValidationError):
    pass


def validate_image_bytes(
    data: bytes,
    *,
    allowed_formats: set[str] | None = None,
    max_file_bytes: int | None = None,
    max_pixels: int | None = None,
    min_dimension: int | None = None,
) -> tuple[Image.Image, dict]:
    """Validate raw uploaded bytes and return ``(rgb_image, info_dict)``.

    Raises a specific :class:`ImageValidationError` subclass for every failure
    mode listed in the project test plan (empty, corrupt, unsupported format,
    too many bytes, too many pixels, too small).
    """
    allowed_formats = allowed_formats or config.ALLOWED_FORMATS
    max_file_bytes = config.MAX_FILE_BYTES if max_file_bytes is None else max_file_bytes
    max_pixels = config.MAX_PIXELS if max_pixels is None else max_pixels
    min_dimension = config.MIN_DIMENSION_PX if min_dimension is None else min_dimension

    # 1. Empty / tiny payloads ------------------------------------------------
    if not data:
        raise EmptyFileError(
            "The uploaded file is empty. Please choose a valid JPG or PNG image."
        )

    # 2. Byte-size guard ------------------------------------------------------
    if len(data) > max_file_bytes:
        raise FileTooLargeError(
            f"The file is {len(data) / (1024 * 1024):.1f} MB, which exceeds the "
            f"{max_file_bytes / (1024 * 1024):.0f} MB limit. Please upload a smaller image."
        )

    # 3. Decode from bytes - the format check uses the file's real content ----
    try:
        img = Image.open(io.BytesIO(data))
        true_format = (img.format or "").upper()
    except UnidentifiedImageError as exc:
        raise CorruptImageError(
            "The file could not be decoded as an image. It may be corrupt, "
            "truncated, or have a non-image extension (e.g. a text file "
            "renamed to .png). Please upload a valid JPG or PNG.",
            detail=str(exc),
        ) from exc
    except DecompressionBombError as exc:
        raise ImageDimensionsTooLargeError(
            "This image declares an enormous size and was rejected as a "
            "safety precaution. Please upload a smaller image.",
            detail=str(exc),
        ) from exc

    # 4. Content-sniffed format check (NOT extension based) -------------------
    if true_format not in allowed_formats:
        raise UnsupportedFormatError(
            f"Unsupported image format '{true_format or 'unknown'}'. VeriVision "
            "accepts JPG/JPEG and PNG files only."
        )

    # 5. Integrity check (catches truncated / corrupt JPEGs and PNGs) ---------
    try:
        img.verify()
    except Exception as exc:  # Pillow raises several exception types here
        raise CorruptImageError(
            "The image appears to be corrupt or truncated and could not be "
            "fully decoded. Try re-exporting the file and uploading it again.",
            detail=str(exc),
        ) from exc

    # Pillow requires a fresh handle after verify() ---------------------------
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except Exception as exc:
        raise CorruptImageError(
            "The image appears to be corrupt or truncated and could not be "
            "fully decoded. Try re-exporting the file and uploading it again.",
            detail=str(exc),
        ) from exc

    width, height = img.size

    # 6. Dimension guards ------------------------------------------------------
    if min(width, height) < min_dimension:
        raise ImageDimensionsTooSmallError(
            f"The image is only {width}x{height} pixels. A minimum side of "
            f"{min_dimension} pixels is required for reliable analysis."
        )
    if width * height > max_pixels:
        raise ImageDimensionsTooLargeError(
            f"The image is {width}x{height} ({width * height / 1e6:.0f} MP), above the "
            f"{max_pixels / 1e6:.0f} MP limit. Please downscale the image and retry."
        )

    # 7. Normalise: EXIF orientation, then RGB --------------------------------
    img = exif_transpose(img)
    rgb = ensure_rgb(img)
    width, height = rgb.size

    info = {
        "declared_format": true_format,
        "original_mode": img.mode,
        "width": width,
        "height": height,
        "file_bytes": len(data),
        "megapixels": round(width * height / 1e6, 2),
    }
    return rgb, info
