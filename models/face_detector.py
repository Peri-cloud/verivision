"""Face detection with two interchangeable OpenCV backends.

1. YuNet (cv2.FaceDetectorYN) - preferred; a small ONNX model (Apache-2.0)
   downloaded once by scripts/download_models.py into .model_cache/yunet/.
2. Haar cascade (haarcascade_frontalface_default.xml) - bundled with
   opencv-python; used automatically when the YuNet weights are absent, so
   the app still works fully offline.

The detector returns raw boxes only. Deciding which boxes are usable and
cropping happens in analysis.py, so this module stays stateless and easy to
unit-test.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

import config


class FaceDetectionError(RuntimeError):
    """No face-detection backend could be initialised."""


@dataclass
class FaceBox:
    x: int
    y: int
    w: int
    h: int
    score: float
    backend: str

    def as_dict(self) -> dict:
        return asdict(self)


class FaceDetector:
    """Detect frontal faces in still images. `detect()` returns ALL boxes;
    filtering by minimum size is the caller's concern."""

    def __init__(self, yunet_weights_path: Path | None = None):
        self._yunet_weights_path = Path(yunet_weights_path or config.YUNET_WEIGHTS_PATH)
        self._yunet = None
        self._haar = None
        self.backend: str | None = None

    @property
    def is_loaded(self) -> bool:
        return self.backend is not None

    def load(self) -> "FaceDetector":
        if self.backend is not None:
            return self

        if self._yunet_weights_path.exists():
            try:
                self._yunet = cv2.FaceDetectorYN.create(
                    str(self._yunet_weights_path),
                    "",
                    (320, 320),
                    score_threshold=config.YUNET_SCORE_THRESHOLD,
                )
                self.backend = "yunet"
                return self
            except cv2.error:
                pass  # fall through to Haar

        cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        if cascade.empty():
            raise FaceDetectionError(
                "Neither the YuNet weights nor the bundled OpenCV Haar cascade "
                "could be loaded. Reinstall opencv-python-headless or run "
                "python scripts/download_models.py --yunet."
            )
        self._haar = cascade
        self.backend = "haar"
        return self

    def detect(self, image: Image.Image) -> list[FaceBox]:
        """Return detected face boxes (in ORIGINAL image coordinates), largest
        first. Detection runs on a downscaled copy (config.DETECT_MAX_SIDE)
        which measurably reduces background false positives; boxes are scaled
        back afterwards. Returns [] for images below the absolute minimum
        working size instead of failing."""
        self.load()
        rgb = np.asarray(image.convert("RGB"))
        height, width = rgb.shape[:2]
        if height < 40 or width < 40:
            return []

        # Downscale for detection, scale boxes back to original coordinates.
        max_side = max(1, int(config.DETECT_MAX_SIDE))
        scale = min(1.0, max_side / max(width, height))
        if scale < 1.0:
            det_w, det_h = int(width * scale), int(height * scale)
            det_rgb = np.asarray(image.convert("RGB").resize((det_w, det_h)))
        else:
            det_w, det_h, det_rgb = width, height, rgb
        bgr = cv2.cvtColor(det_rgb, cv2.COLOR_RGB2BGR)

        if self.backend == "yunet":
            self._yunet.setInputSize((det_w, det_h))
            _, faces = self._yunet.detect(bgr)
            if faces is None:
                return []
            inv = 1.0 / scale
            boxes = [
                FaceBox(
                    x=int(f[0] * inv), y=int(f[1] * inv),
                    w=int(f[2] * inv), h=int(f[3] * inv),
                    score=float(f[-1]), backend="yunet",
                )
                for f in faces
            ]
        else:
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            rects = self._haar.detectMultiScale(
                gray,
                scaleFactor=config.HAAR_SCALE_FACTOR,
                minNeighbors=config.HAAR_MIN_NEIGHBOURS,
                minSize=(max(24, int(60 * scale)), max(24, int(60 * scale))),
            )
            inv = 1.0 / scale
            boxes = [
                FaceBox(x=int(x * inv), y=int(y * inv), w=int(w * inv), h=int(h * inv),
                        score=1.0, backend="haar")
                for (x, y, w, h) in rects
            ]

        boxes.sort(key=lambda b: b.w * b.h, reverse=True)
        return boxes


def crop_face(image: Image.Image, box: FaceBox, margin: float | None = None) -> Image.Image:
    """Crop a face region with a configurable relative margin, clamped to the
    image bounds."""
    margin = config.FACE_CROP_MARGIN if margin is None else margin
    left, top = max(0, int(box.x - box.w * margin)), max(0, int(box.y - box.h * margin))
    right = min(image.width, int(box.x + box.w * (1 + margin)))
    bottom = min(image.height, int(box.y + box.h * (1 + margin)))
    return image.crop((left, top, right, bottom))


def draw_boxes(image: Image.Image, boxes: list[FaceBox], labels: list[str] | None = None) -> Image.Image:
    """Return an annotated copy of `image` with red boxes (and optional index
    labels) drawn on it. Uses PIL so no BGR/RGB confusion can occur."""
    out = image.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    for i, box in enumerate(boxes):
        draw.rectangle(
            (box.x, box.y, box.x + box.w, box.y + box.h),
            outline=(220, 30, 30),
            width=max(2, out.width // 400),
        )
        if labels is not None and i < len(labels):
            draw.text((box.x + 4, max(0, box.y - 14)), labels[i], fill=(220, 30, 30))
    return out
