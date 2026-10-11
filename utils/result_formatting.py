"""Interpretation of raw model outputs into human-readable, honestly
qualified results.

DESIGN NOTE ON THRESHOLDS
-------------------------
Both detectors are two-class classifiers whose softmax outputs are NOT
calibrated probabilities. The decision boundary 0.5 is simply the argmax
boundary; scores inside +/- UNCERTAINTY_BAND of the boundary are reported as
"Inconclusive". The band is a conservative interface heuristic documented in
the README - it is not a validated calibration procedure, and no result in
this app should be read as a calibrated probability of ground truth.
"""
from __future__ import annotations

import enum

import config
from models.common import RawPrediction


class Verdict(str, enum.Enum):
    """Accessible status labels (never colour-only: the UI always shows text)."""

    REAL = "Likely real photograph"
    AI_GENERATED = "Likely AI-generated image"
    MANIPULATED = "Likely manipulated / deepfake"
    INCONCLUSIVE = "Inconclusive"
    NO_FACE = "No face detected"
    CONFLICT = "Conflicting predictions"
    ERROR = "Analysis error"


# (bootstrap colour class, short explanation) used by the UI layer
VERDICT_STYLES: dict[Verdict, dict] = {
    Verdict.REAL: {"tone": "positive", "note": "The detector found no evidence of synthesis."},
    Verdict.AI_GENERATED: {"tone": "negative", "note": "The detector scored this image as synthetic."},
    Verdict.MANIPULATED: {"tone": "negative", "note": "The detector scored this face as manipulated."},
    Verdict.INCONCLUSIVE: {"tone": "neutral", "note": "The score falls inside the documented uncertainty band."},
    Verdict.NO_FACE: {"tone": "neutral", "note": "No usable face was found, so the deepfake detector was not applied."},
    Verdict.CONFLICT: {"tone": "warning", "note": "Faces in this image received different verdicts - see each face below."},
    Verdict.ERROR: {"tone": "neutral", "note": "The detector could not be applied to this input."},
}


def _band_verdict(fake_score: float) -> str:
    """Classify a fake-class score as 'fake' | 'uncertain' | 'real'."""
    band = max(0.0, float(config.UNCERTAINTY_BAND))
    boundary = float(config.DECISION_BOUNDARY)
    if band > 0 and abs(fake_score - boundary) < band:
        return "uncertain"
    return "fake" if fake_score >= boundary else "real"


def interpret_ai_image_score(raw: RawPrediction) -> dict:
    """Map Model A's raw output to a verdict dict (whole-image AI detection)."""
    mapping = {
        "real": Verdict.REAL,
        "fake": Verdict.AI_GENERATED,
        "uncertain": Verdict.INCONCLUSIVE,
    }
    return _format_result(raw, mapping, model_kind="AI-image")


def interpret_deepfake_score(raw: RawPrediction) -> dict:
    """Map Model B's raw output to a verdict dict (per-face deepfake)."""
    mapping = {
        "real": Verdict.REAL,
        "fake": Verdict.MANIPULATED,
        "uncertain": Verdict.INCONCLUSIVE,
    }
    return _format_result(raw, mapping, model_kind="deepfake")


def _format_result(raw: RawPrediction, mapping: dict[str, Verdict], *, model_kind: str) -> dict:
    band = max(0.0, float(config.UNCERTAINTY_BAND))
    used_band = (
        band > 0 and abs(raw.fake_score - float(config.DECISION_BOUNDARY)) < band
    )
    verdict = mapping[_band_verdict(raw.fake_score)]
    return {
        "model_kind": model_kind,
        "verdict": verdict,
        "fake_score": round(raw.fake_score, 4),
        "probabilities": {k: round(v, 4) for k, v in raw.probabilities.items()},
        "used_uncertainty_band": used_band,
        "score_description": describe_score(model_kind),
        "model_id": raw.model_id,
        "device": raw.device,
        "inference_ms": round(raw.inference_ms, 1),
        "warnings": list(raw.warnings),
    }


def summarize_faces(face_results: list[dict]) -> dict:
    """Summarise per-face deepfake results without hiding disagreement.

    If every face agrees, that verdict wins. If ANY two faces disagree, the
    summary is Verdict.CONFLICT with per-verdict counts - never an average
    score, which would hide the disagreement.
    """
    if not face_results:
        return {
            "verdict": Verdict.NO_FACE,
            "face_count": 0,
            "counts": {},
            "narrative": (
                "No face was detected in this image, so the deepfake detector "
                "was not applied. No deepfake score is reported because the "
                "model is trained for facial manipulation only."
            ),
            "conflicting": False,
        }

    verdicts = [fr["verdict"] for fr in face_results]
    counts: dict[str, int] = {}
    for v in verdicts:
        counts[v.value] = counts.get(v.value, 0) + 1
    unique = set(verdicts)

    if len(unique) == 1:
        verdict = verdicts[0]
        narrative = f"All {len(verdicts)} detected face(s) received the same verdict: {verdict.value}."
        conflicting = False
    else:
        verdict = Verdict.CONFLICT
        conflicting = True
        # counts maps "verdict label -> occurrences"; keys are already plain strings
        parts = [f"{n}x {v}" for v, n in counts.items()]
        narrative = (
            f"Faces received conflicting verdicts ({', '.join(parts)}). "
            "Review each face individually below - the disagreement itself is "
            "a signal that automated classification is unreliable here."
        )

    return {
        "verdict": verdict,
        "face_count": len(face_results),
        "counts": counts,
        "narrative": narrative,
        "conflicting": conflicting,
    }


def describe_score(model_kind: str) -> str:
    """Honest description of what the displayed score means."""
    if model_kind == "AI-image":
        return (
            "Score = raw softmax output of the 'artificial' class, where 0.00 "
            "leans real and 1.00 leans synthetic. Values near 0.5 are ambiguous. "
            "This is NOT a calibrated probability: the checkpoint ships without "
            "calibration, so the number describes the model's output, not the "
            "true likelihood that the image is fake."
        )
    return (
        "Score = sigmoid of the deepfake detector's single output logit, as "
        "defined by the model card, where 0.00 leans real and 1.00 leans "
        "manipulated. Values near 0.5 are ambiguous. This is NOT a calibrated "
        "probability: the number describes the model's output, not the true "
        "likelihood of manipulation."
    )


def format_duration(ms: float) -> str:
    if ms < 1000:
        return f"{ms:.0f} ms"
    return f"{ms / 1000:.2f} s"
