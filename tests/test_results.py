"""Verdict-mapping, uncertainty-band and multi-face summary tests."""
from __future__ import annotations

import pytest

import config
from models.common import RawPrediction
from utils.result_formatting import (
    Verdict,
    describe_score,
    interpret_ai_image_score,
    interpret_deepfake_score,
    summarize_faces,
)


def raw(score: float) -> RawPrediction:
    return RawPrediction(
        probabilities={"real": 1 - score, "fake": score},
        fake_score=score,
        model_id="mock/model",
        device="cpu",
        inference_ms=5.0,
    )


class TestAIVerdicts:
    def test_high_score_is_ai_generated(self):
        assert interpret_ai_image_score(raw(0.98))["verdict"] is Verdict.AI_GENERATED

    def test_low_score_is_real(self):
        assert interpret_ai_image_score(raw(0.02))["verdict"] is Verdict.REAL

    def test_boundary_score_inside_band_is_inconclusive(self):
        assert interpret_ai_image_score(raw(0.52))["verdict"] is Verdict.INCONCLUSIVE
        assert interpret_ai_image_score(raw(0.52))["used_uncertainty_band"] is True

    def test_band_edge_is_inconclusive_but_beyond_is_not(self):
        # band 0.10 -> (0.40, 0.60) uncertain by default config
        assert interpret_ai_image_score(raw(0.41))["verdict"] is Verdict.INCONCLUSIVE
        assert interpret_ai_image_score(raw(0.61))["verdict"] is Verdict.AI_GENERATED
        assert interpret_ai_image_score(raw(0.39))["verdict"] is Verdict.REAL

    def test_band_can_be_disabled(self, monkeypatch):
        monkeypatch.setattr(config, "UNCERTAINTY_BAND", 0.0)
        res = interpret_ai_image_score(raw(0.52))
        assert res["verdict"] is Verdict.AI_GENERATED
        assert res["used_uncertainty_band"] is False


class TestDeepfakeVerdicts:
    def test_high_score_is_manipulated(self):
        assert interpret_deepfake_score(raw(0.95))["verdict"] is Verdict.MANIPULATED

    def test_low_score_is_real(self):
        assert interpret_deepfake_score(raw(0.05))["verdict"] is Verdict.REAL

    def test_uncertain(self):
        assert interpret_deepfake_score(raw(0.5))["verdict"] is Verdict.INCONCLUSIVE

    def test_score_description_never_claims_calibration(self):
        text = describe_score("deepfake")
        assert "NOT a calibrated probability" in text


class TestFaceSummaries:
    def _face(self, verdict: Verdict, index: int = 1) -> dict:
        return {"index": index, "verdict": verdict, "fake_score": 0.1}

    def test_no_faces_gives_no_face_without_score(self):
        summary = summarize_faces([])
        assert summary["verdict"] is Verdict.NO_FACE
        assert summary["face_count"] == 0

    def test_all_real(self):
        summary = summarize_faces([self._face(Verdict.REAL), self._face(Verdict.REAL, 2)])
        assert summary["verdict"] is Verdict.REAL
        assert summary["conflicting"] is False

    def test_all_manipulated(self):
        summary = summarize_faces([self._face(Verdict.MANIPULATED)])
        assert summary["verdict"] is Verdict.MANIPULATED

    def test_mixed_predictions_surface_conflict(self):
        summary = summarize_faces(
            [self._face(Verdict.REAL), self._face(Verdict.MANIPULATED, 2)]
        )
        assert summary["verdict"] is Verdict.CONFLICT
        assert summary["conflicting"] is True
        assert summary["counts"] == {
            Verdict.REAL.value: 1,
            Verdict.MANIPULATED.value: 1,
        }

    def test_inconclusive_alone(self):
        summary = summarize_faces([self._face(Verdict.INCONCLUSIVE)])
        assert summary["verdict"] is Verdict.INCONCLUSIVE


class TestLabelMappingSemantics:
    def test_configured_output_modes(self):
        """Documents verified semantics: Model A is softmax with the fake class
        at index 0; Model B is a single sigmoid logit (CommunityForensics)."""
        assert config.AI_IMAGE_OUTPUT_MODE == "softmax"
        assert config.AI_IMAGE_FAKE_CLASS_INDEX == 0
        assert config.AI_IMAGE_EXPECTED_LABELS == {"artificial", "real"}
        assert config.DEEPFAKE_OUTPUT_MODE == "sigmoid"
        assert config.DEEPFAKE_EXPECTED_LABELS == {"fake", "real"}
