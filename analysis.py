"""UI-agnostic analysis orchestration.

`app.py` (Streamlit) only wires cached detector singletons into these pure
functions, which keeps every decision - verdict mapping, multi-face
summarisation, small-face filtering, error isolation - unit-testable without
Streamlit or network access.
"""
from __future__ import annotations

import time

import config
from models.ai_image_detector import AIImageDetector
from models.common import InferenceError, ModelLoadError
from models.deepfake_detector import DeepfakeDetector
from models.face_detector import FaceBox, FaceDetector, crop_face, draw_boxes
from utils.result_formatting import (
    Verdict,
    interpret_ai_image_score,
    interpret_deepfake_score,
    summarize_faces,
)


def run_ai_image_check(image, detector: AIImageDetector) -> dict:
    """Whole-image AI-generation check. Never raises for model problems:
    failures become an ERROR-verdict result the UI can render."""
    start = time.perf_counter()
    try:
        raw = detector.predict(image)
    except ModelLoadError as exc:
        return _error_result("AI-image", exc, start, load_failure=True)
    except InferenceError as exc:
        return _error_result("AI-image", exc, start)
    result = interpret_ai_image_score(raw)
    result["total_ms"] = round((time.perf_counter() - start) * 1000, 1)
    return result


def run_deepfake_check(
    image,
    face_detector: FaceDetector,
    deepfake_detector: DeepfakeDetector,
) -> dict:
    """Full face pipeline: detect faces -> crop -> per-face inference ->
    summary. Key guarantees:

    * No face  -> verdict NO_FACE and NO fabricated score.
    * Faces too small to trust -> reported as a limitation, not a score.
    * Each face keeps its own verdict; conflicting predictions are surfaced,
      never averaged away.
    * A model failure on face N does not poison faces 1..N-1.
    """
    start = time.perf_counter()
    limitations: list[str] = []

    try:
        boxes = face_detector.detect(image)
    except Exception as exc:  # noqa: BLE001 - any detector failure becomes an ERROR result
        return {
            "faces": [],
            "summary": {
                "verdict": Verdict.ERROR,
                "face_count": 0,
                "counts": {},
                "narrative": f"Face detection failed: {exc}",
                "conflicting": False,
            },
            "face_count": 0,
            "annotated_image": None,
            "limitations": [f"Face detection failed: {exc}"],
            "total_ms": round((time.perf_counter() - start) * 1000, 1),
        }

    if not boxes:
        summary = summarize_faces([])
        summary["narrative"] += (
            " If you expected a face, try a larger, well-lit, frontal photo - "
            "heavy occlusion, profile angles or very small faces are common causes."
        )
        return {
            "faces": [],
            "summary": summary,
            "face_count": 0,
            "annotated_image": None,
            "limitations": ["No face detected - the deepfake model was not run."],
            "total_ms": round((time.perf_counter() - start) * 1000, 1),
        }

    usable: list[FaceBox] = []
    small: list[FaceBox] = []
    for box in boxes:
        (usable if box.w >= config.MIN_FACE_SIZE_PX and box.h >= config.MIN_FACE_SIZE_PX else small).append(box)

    if small:
        limitations.append(
            f"{len(small)} detected face region(s) were below the "
            f"{config.MIN_FACE_SIZE_PX}px minimum and were skipped (too small to "
            "analyse reliably)."
        )
    if len(usable) > config.MAX_FACES:
        limitations.append(
            f"{len(usable)} faces detected; analysing the {config.MAX_FACES} largest."
        )
        usable = usable[: config.MAX_FACES]

    if not usable:
        return {
            "faces": [],
            "summary": summarize_faces([]),
            "face_count": 0,
            "annotated_image": draw_boxes(image, boxes, [f"{i + 1}" for i in range(len(boxes))]),
            "limitations": limitations,
            "total_ms": round((time.perf_counter() - start) * 1000, 1),
        }

    face_results: list[dict] = []
    for i, box in enumerate(usable):
        crop = crop_face(image, box)
        try:
            raw = deepfake_detector.predict(crop)
        except (ModelLoadError, InferenceError) as exc:
            face_results.append(
                {
                    "index": i + 1,
                    "box": box.as_dict(),
                    "verdict": Verdict.ERROR,
                    "fake_score": None,
                    "probabilities": {},
                    "score_description": "",
                    "error_message": str(exc),
                    "crop": crop,
                    "inference_ms": None,
                    "warnings": [],
                }
            )
            continue
        result = interpret_deepfake_score(raw)
        result.update({"index": i + 1, "box": box.as_dict(), "crop": crop})
        face_results.append(result)

    summary = summarize_faces(face_results)
    return {
        "faces": face_results,
        "summary": summary,
        "face_count": len(face_results),
        "annotated_image": draw_boxes(image, boxes, [f"Face {i + 1}" for i in range(len(usable))]),
        "limitations": limitations,
        "total_ms": round((time.perf_counter() - start) * 1000, 1),
    }


def run_full_analysis(
    image,
    *,
    ai_detector: AIImageDetector | None,
    face_detector: FaceDetector | None,
    deepfake_detector: DeepfakeDetector | None,
    mode: str,
) -> dict:
    """Dispatch one of the three analysis modes. Both detectors in 'both'
    mode produce INDEPENDENT result dicts - neither can overwrite the other."""
    start = time.perf_counter()
    output: dict = {"mode": mode, "ai_image": None, "deepfake": None}

    if mode in ("AI-Generated Image Detection", "Both Detectors") and ai_detector is not None:
        output["ai_image"] = run_ai_image_check(image, ai_detector)

    if mode in ("Deepfake Face Detection", "Both Detectors") and face_detector is not None and deepfake_detector is not None:
        output["deepfake"] = run_deepfake_check(image, face_detector, deepfake_detector)

    output["total_ms"] = round((time.perf_counter() - start) * 1000, 1)
    return output


def _error_result(kind: str, exc: Exception, start: float, *, load_failure: bool = False) -> dict:
    return {
        "model_kind": kind,
        "verdict": Verdict.ERROR,
        "fake_score": None,
        "probabilities": {},
        "used_uncertainty_band": False,
        "score_description": "",
        "error_message": str(exc),
        "load_failure": load_failure,
        "model_id": config.AI_IMAGE_MODEL_ID if kind == "AI-image" else config.DEEPFAKE_MODEL_ID,
        "device": None,
        "inference_ms": None,
        "warnings": [],
        "total_ms": round((time.perf_counter() - start) * 1000, 1),
    }
