"""Real-inference smoke test (no mocks, no Streamlit).

Runs BOTH detectors over the bundled sample images and prints a transparent
report. By default it REPORTS the verdicts without asserting them - models
may legitimately disagree with our expectations on out-of-domain data - and
fails only if a model cannot be loaded or inference raises.

Use --strict to additionally assert the expected verdicts for the bundled
samples (this is how development validated the label mapping).

Usage:
    python scripts/smoke_test.py [--strict]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from analysis import run_full_analysis
from utils.image_validation import ImageValidationError, validate_image_bytes

SAMPLES = config.SAMPLE_IMAGES_DIR

# (filename, expected verdict when --strict) - expectations were measured on
# the development machine and are deliberately conservative.
AI_CASES = [
    ("sample_real_scene.jpg", "real"),
    ("sample_ai_scene.jpg", "fake"),
    ("sample_ai_food.jpg", "fake"),
]
FACE_CASES = [
    ("sample_real_portrait.jpg", "real"),
    ("sample_ai_portrait.jpg", None),  # synthetic face; model may disagree
]
NO_FACE_CASES = [("sample_ai_food.jpg", None)]


def load(name: str):
    return validate_image_bytes((SAMPLES / name).read_bytes())[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    failures: list[str] = []
    print(f"Sample images: {SAMPLES}")
    print(f"AI-image model : {config.AI_IMAGE_MODEL_ID}")
    print(f"Deepfake model : {config.DEEPFAKE_MODEL_ID}\n")

    # ---------------- AI-image detector ----------------
    try:
        from models.ai_image_detector import AIImageDetector

        ai = AIImageDetector().load()
        for name, expected in AI_CASES:
            image = load(name)
            t0 = time.perf_counter()
            result = run_full_analysis(image, ai_detector=ai, face_detector=None,
                                       deepfake_detector=None, mode="AI-Generated Image Detection")
            res = result["ai_image"]
            verdict = res["verdict"].value
            print(f"[AI-image] {name:<26} fake_score={res['fake_score']:.4f} "
                  f"-> {verdict} ({res['total_ms']:.0f} ms)")
            if args.strict and expected and res["verdict"].value.lower() != ("likely ai-generated image" if expected == "fake" else "likely real photograph"):
                failures.append(f"AI-image {name}: expected {expected}, got {verdict}")
    except Exception as exc:  # noqa: BLE001
        print(f"[AI-image] FAILED: {type(exc).__name__}: {exc}")
        failures.append(f"AI-image detector crashed: {exc}")

    # ---------------- Deepfake pipeline ----------------
    try:
        from models.deepfake_detector import DeepfakeDetector
        from models.face_detector import FaceDetector

        face_det = FaceDetector().load()
        deep = DeepfakeDetector().load()
        print(f"Face backend: {face_det.backend}")
        for name, expected in FACE_CASES:
            image = load(name)
            result = run_full_analysis(image, ai_detector=None, face_detector=face_det,
                                       deepfake_detector=deep, mode="Deepfake Face Detection")
            res = result["deepfake"]
            summary = res["summary"]
            print(f"[deepfake] {name:<26} faces={res['face_count']} "
                  f"-> {summary['verdict'].value}")
            for face in res["faces"]:
                print(f"           face {face['index']}: score={face['fake_score']} "
                      f"-> {face['verdict'].value}")
            if args.strict and expected and summary["verdict"].value != "Likely real photograph":
                failures.append(f"deepfake {name}: expected real, got {summary['verdict'].value}")

        # No-face behaviour must never fabricate a score
        image = load(NO_FACE_CASES[0][0])
        result = run_full_analysis(image, ai_detector=None, face_detector=face_det,
                                   deepfake_detector=deep, mode="Deepfake Face Detection")
        res = result["deepfake"]
        no_score = all(f.get("fake_score") is None for f in res["faces"])
        print(f"[deepfake] {NO_FACE_CASES[0][0]:<26} faces={res['face_count']} "
              f"-> {res['summary']['verdict'].value} (no fabricated score: {no_score})")
        if res["face_count"] != 0 or not no_score:
            failures.append("no-face image produced a face result")
    except Exception as exc:  # noqa: BLE001
        print(f"[deepfake] FAILED: {type(exc).__name__}: {exc}")
        failures.append(f"deepfake pipeline crashed: {exc}")

    print()
    if failures:
        print("STRICT FAILURES:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("Smoke test finished. Detector exceptions: none. "
          "(Verdicts are reported as-is; use --strict to assert them.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
