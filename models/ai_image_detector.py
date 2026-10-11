"""Model A wrapper - AI-generated image detection (whole-image).

Model: haywoodsloan/ai-image-detector-deploy (SwinV2, 256x256, Apache-2.0)
Verified label mapping: id2label = {0: 'artificial', 1: 'real'}, so the FAKE
class index is 0. The mapping is re-validated on every load (see common.py).

This detector is trained to distinguish real photographs from AI-GENERATED
IMAGES. It is NOT a deepfake (face-manipulation) detector - the two tasks use
separate models on purpose. See the README for its documented limitations
(AutoTrain dataset is not fully disclosed; benchmark numbers are in-domain).
"""
from __future__ import annotations

import config
from models.common import (
    InferenceError,
    RawPrediction,
    load_image_classifier,
    resolve_device,
    run_classifier,
)


class AIImageDetector:
    """Lazy-loading wrapper so the ~780 MB checkpoint is only materialised
    when the AI-image mode (or Both mode) actually needs it."""

    MODEL_ID = config.AI_IMAGE_MODEL_ID
    KIND = "AI-image"

    def __init__(self, device: str | None = None):
        self.device = device or resolve_device()
        self._processor = None
        self._model = None
        self.id2label: dict[int, str] | None = None
        self.fake_class_index = config.AI_IMAGE_FAKE_CLASS_INDEX

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> "AIImageDetector":
        if self._model is None:
            self._processor, self._model, self.id2label = load_image_classifier(
                self.MODEL_ID,
                expected_labels=config.AI_IMAGE_EXPECTED_LABELS,
                fake_class_index=self.fake_class_index,
                device=self.device,
            )
        return self

    def predict(self, image) -> RawPrediction:
        """Run whole-image inference. `image` is any PIL image."""
        self.load()
        try:
            return run_classifier(
                self._processor,
                self._model,
                self.device,
                image,
                fake_class_index=self.fake_class_index,
                model_id=self.MODEL_ID,
            )
        except InferenceError:
            if self.device != "cpu":
                # Graceful accelerator fallback (e.g. an unsupported op on
                # Apple MPS): retry once on CPU before giving up.
                warnings = [f"Device '{self.device}' failed; retried on CPU."]
                self.device = "cpu"
                self._model.to("cpu")
                return run_classifier(
                    self._processor,
                    self._model,
                    "cpu",
                    image,
                    fake_class_index=self.fake_class_index,
                    model_id=self.MODEL_ID,
                    extra_warnings=warnings,
                )
            raise
