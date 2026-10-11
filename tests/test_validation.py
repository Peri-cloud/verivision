"""Upload-validation tests: every failure mode from the project test plan."""
from __future__ import annotations

import io

import pytest
from PIL import Image

import config
from utils.image_validation import (
    CorruptImageError,
    EmptyFileError,
    FileTooLargeError,
    ImageDimensionsTooLargeError,
    ImageDimensionsTooSmallError,
    UnsupportedFormatError,
    validate_image_bytes,
)


class TestValidUploads:
    def test_valid_jpeg_upload(self, valid_jpeg):
        img, info = validate_image_bytes(valid_jpeg)
        assert img.mode == "RGB"
        assert info["declared_format"] == "JPEG"
        assert info["width"] == 320 and info["height"] == 240

    def test_valid_png_upload(self, valid_png):
        img, info = validate_image_bytes(valid_png)
        assert img.mode == "RGB"
        assert info["declared_format"] == "PNG"

    def test_rgba_png_becomes_rgb(self, rgba_png):
        img, info = validate_image_bytes(rgba_png)
        assert img.mode == "RGB"
        assert info["original_mode"] == "RGBA"

    def test_exif_orientation_applied(self, exif_rotated_jpeg):
        """Orientation 6 rotates the 100x60 source to 60x100 when applied."""
        img, info = validate_image_bytes(exif_rotated_jpeg)
        assert (info["width"], info["height"]) == (60, 100)


class TestCorruptAndUnsupported:
    def test_empty_upload_rejected(self):
        with pytest.raises(EmptyFileError):
            validate_image_bytes(b"")

    def test_truncated_jpeg_rejected(self, corrupt_jpeg):
        with pytest.raises(CorruptImageError):
            validate_image_bytes(corrupt_jpeg)

    def test_text_renamed_to_png_rejected(self, text_renamed_png):
        with pytest.raises(CorruptImageError):
            validate_image_bytes(text_renamed_png)

    def test_unsupported_format_rejected(self, bmp_bytes):
        with pytest.raises(UnsupportedFormatError):
            validate_image_bytes(bmp_bytes)


class TestSizeLimits:
    def test_file_bytes_limit(self, valid_jpeg):
        with pytest.raises(FileTooLargeError):
            validate_image_bytes(valid_jpeg, max_file_bytes=10)

    def test_pixel_limit(self, valid_jpeg):
        with pytest.raises(ImageDimensionsTooLargeError):
            validate_image_bytes(valid_jpeg, max_pixels=100)

    def test_too_small_rejected(self, valid_jpeg):
        with pytest.raises(ImageDimensionsTooSmallError):
            validate_image_bytes(valid_jpeg, min_dimension=500)

    def test_pillow_bomb_guard(self, monkeypatch):
        """A 1200x1200 image (1.44 MP) must fail when the pixel limit is 1 MP."""
        monkeypatch.setattr(config, "MAX_PIXELS", 1_000_000)
        img = Image.new("RGB", (1200, 1200))  # 1,440,000 pixels - cheap to build
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        with pytest.raises(ImageDimensionsTooLargeError):
            validate_image_bytes(buf.getvalue())


class TestDefaultsFromConfig:
    def test_limits_come_from_config(self, valid_jpeg):
        assert config.MAX_FILE_BYTES == 25 * 1024 * 1024
        # defaults should accept a normal 320x240 image
        img, _ = validate_image_bytes(valid_jpeg)
        assert img is not None
