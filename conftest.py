"""Shared pytest fixtures.

Mock model outputs appear ONLY here (and in tests) - the application itself
never falls back to a dummy predictor.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402
from models.common import RawPrediction  # noqa: E402
from models.face_detector import FaceBox  # noqa: E402


# ----------------------------------------------------------------- images
def make_image(w: int = 320, h: int = 240, color=(120, 130, 140)) -> Image.Image:
    return Image.new("RGB", (w, h), color)


def jpeg_bytes(img: Image.Image | None = None) -> bytes:
    img = img or make_image()
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def png_bytes(img: Image.Image | None = None) -> bytes:
    img = img or make_image()
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def valid_jpeg() -> bytes:
    return jpeg_bytes()


@pytest.fixture
def valid_png() -> bytes:
    return png_bytes()


@pytest.fixture
def corrupt_jpeg() -> bytes:
    """A real JPEG with its tail sliced off (truncated mid-stream)."""
    data = jpeg_bytes()
    return data[: len(data) // 2]


@pytest.fixture
def text_renamed_png() -> bytes:
    """Text content pretending to be a PNG (extension-vs-content mismatch)."""
    return b"this is definitely not an image, despite any file extension."


@pytest.fixture
def bmp_bytes() -> bytes:
    buf = io.BytesIO()
    make_image().save(buf, format="BMP")
    return buf.getvalue()


@pytest.fixture
def rgba_png() -> bytes:
    img = Image.new("RGBA", (160, 120), (200, 30, 30, 128))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def exif_rotated_jpeg() -> bytes:
    """JPEG carrying EXIF orientation 6 (rotate 90 deg CW on display)."""
    img = Image.new("RGB", (100, 60), (10, 90, 200))
    exif = Image.Exif()
    exif[274] = 6  # 0x0112 Orientation
    buf = io.BytesIO()
    img.save(buf, format="JPEG", exif=exif)
    return buf.getvalue()


# ------------------------------------------------------- mocked detectors
def fake_raw(fake_score: float, probs: dict | None = None) -> RawPrediction:
    probabilities = probs or {"real": 1.0 - fake_score, "artificial": fake_score}
    return RawPrediction(
        probabilities=probabilities,
        fake_score=fake_score,
        model_id="mock/model",
        device="cpu",
        inference_ms=1.0,
    )


class FakeAIDetector:
    """Deterministic stand-in for AIImageDetector (tests only)."""

    KIND = "AI-image"

    def __init__(self, fake_score: float):
        self.fake_score = fake_score

    def predict(self, image):
        return fake_raw(self.fake_score)


class FakeDeepfakeDetector:
    KIND = "deepfake"

    def __init__(self, fake_scores: list[float] | float):
        self.fake_scores = fake_scores if isinstance(fake_scores, list) else [fake_scores]
        self._calls = 0

    def predict(self, image):
        score = self.fake_scores[min(self._calls, len(self.fake_scores) - 1)]
        self._calls += 1
        return fake_raw(score, probs={"Real": 1.0 - score, "Fake": score})


class FakeFaceDetector:
    """Returns preset boxes without running OpenCV."""

    def __init__(self, boxes: list[FaceBox]):
        self.boxes = boxes

    def detect(self, image):
        return self.boxes


@pytest.fixture
def face_box() -> FaceBox:
    return FaceBox(x=100, y=80, w=140, h=140, score=0.97, backend="mock")


@pytest.fixture
def small_face_box() -> FaceBox:
    return FaceBox(x=10, y=10, w=30, h=30, score=0.9, backend="mock")
