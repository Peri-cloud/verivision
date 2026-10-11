"""Model B wrapper - deepfake face detection (per-face crop).

Model: buildborderless/CommunityForensics-DeepfakeDet-ViT
(ViT-S/16, 384x384, MIT, arXiv:2411.04125 - Community Forensics).
Output: a SINGLE sigmoid logit where 1.0 = fake, semantics defined by the
official model card and structurally validated at load time.

Documented behaviour / limitations (see README):
  * trained on 2.7M samples across 4,803 generators for cross-generator
    generalization, but novel generators can still be missed (we measured a
    synthetic face passing as real with score 0.09 - reported, not hidden);
  * it is a FACE-domain detector - it is applied only to face crops, and no
    score is produced when no face is present.
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


class DeepfakeDetector:
    """Lazy-loading wrapper for the deepfake face classifier."""

    MODEL_ID = config.DEEPFAKE_MODEL_ID
    KIND = "deepfake"

    def __init__(self, device: str | None = None):
        self.device = device or resolve_device()
        self._processor = None
        self._model = None
        self.id2label: dict[int, str] | None = None
        self.fake_class_index = config.DEEPFAKE_FAKE_CLASS_INDEX
        self.output_mode = config.DEEPFAKE_OUTPUT_MODE

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> "DeepfakeDetector":
        if self._model is None:
            self._processor, self._model, self.id2label = load_image_classifier(
                self.MODEL_ID,
                expected_labels=config.DEEPFAKE_EXPECTED_LABELS,
                fake_class_index=self.fake_class_index,
                device=self.device,
                output_mode=self.output_mode,
            )
        return self

    def predict(self, face_crop) -> RawPrediction:
        """Run inference on ONE face crop (a PIL image)."""
        self.load()
        try:
            return run_classifier(
                self._processor,
                self._model,
                self.device,
                face_crop,
                fake_class_index=self.fake_class_index,
                model_id=self.MODEL_ID,
                output_mode=self.output_mode,
            )
        except InferenceError:
            if self.device != "cpu":
                warnings = [f"Device '{self.device}' failed; retried on CPU."]
                self.device = "cpu"
                self._model.to("cpu")
                return run_classifier(
                    self._processor,
                    self._model,
                    "cpu",
                    face_crop,
                    fake_class_index=self.fake_class_index,
                    model_id=self.MODEL_ID,
                    output_mode=self.output_mode,
                    extra_warnings=warnings,
                )
            raise
