"""Optional scientific evaluation on a LABELLED dataset you provide.

Expected layout (both directories required, one class each):

    my_dataset/
        real/   *.jpg *.jpeg *.png   -> ground-truth authentic images
        fake/   *.jpg *.jpeg *.png   -> ground-truth AI-generated (ai-image
                                        detector) or manipulated-face
                                        (deepfake detector) images

For the deepfake detector, `real`/`fake` must be FACE images; face-free
images are counted as skipped and excluded from the metrics (never silently
scored).

Measured metrics: accuracy, precision, recall, F1 (fake = positive class),
confusion matrix and ROC-AUC at the documented 0.5 argmax boundary.
Nothing is fabricated: images that error are reported as errors, and AUC is
skipped when a class is missing.

Usage:
    python scripts/evaluate.py --detector ai-image \
        --real-dir data/eval/real --fake-dir data/eval/ai [--limit 50] \
        [--output results/ai_image_eval.json]

IMPORTANT: keep near-duplicates and video frames of the same source in the
same split; never mix them across the real/fake folders or your numbers will
be inflated (see README, evaluation methodology).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from models.ai_image_detector import AIImageDetector
from models.common import InferenceError, ModelLoadError
from models.deepfake_detector import DeepfakeDetector
from models.face_detector import FaceDetector
from utils.image_validation import ImageValidationError, validate_image_bytes

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def collect(folder: Path, limit: int) -> list[Path]:
    files = sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    return files[:limit] if limit > 0 else files


def build_predictor(detector_name: str):
    """Return predict(image) -> fake-score or None (skip). Loads the models."""
    if detector_name == "ai-image":
        detector = AIImageDetector().load()
        return lambda img: detector.predict(img).fake_score

    face_detector = FaceDetector().load()
    deepfake = DeepfakeDetector().load()

    def predict(img):
        boxes = [
            b for b in face_detector.detect(img)
            if b.w >= config.MIN_FACE_SIZE_PX and b.h >= config.MIN_FACE_SIZE_PX
        ]
        if not boxes:
            return None  # no face -> skipped, never scored
        return max(
            deepfake.predict(
                img.crop((b.x, b.y, b.x + b.w, b.y + b.h))
            ).fake_score
            for b in boxes[: config.MAX_FACES]
        )

    return predict


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detector", choices=["ai-image", "deepfake"], required=True)
    parser.add_argument("--real-dir", type=Path, required=True)
    parser.add_argument("--fake-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0, help="max images per class (0 = all)")
    parser.add_argument("--output", type=Path, default=None, help="write JSON report here")
    args = parser.parse_args()

    real_files = collect(args.real_dir, args.limit)
    fake_files = collect(args.fake_dir, args.limit)
    if not real_files or not fake_files:
        print("Both --real-dir and --fake-dir must contain images.", file=sys.stderr)
        return 2

    dataset = [(p, 0) for p in real_files] + [(p, 1) for p in fake_files]
    print(f"Detector: {args.detector} | real: {len(real_files)} | fake: {len(fake_files)} "
          f"| total: {len(dataset)} images")

    predict = build_predictor(args.detector)

    y_true: list[int] = []
    y_score: list[float] = []
    per_file: list[dict] = []
    errors: list[dict] = []
    skipped = 0
    start = time.perf_counter()
    for i, (path, label) in enumerate(dataset):
        try:
            data = path.read_bytes()
            image = validate_image_bytes(data, max_file_bytes=100 * 1024 * 1024)[0]
            score = predict(image)
            if score is None:
                skipped += 1
                print(f"  - {path.name}: skipped (no usable face)")
                continue
            y_true.append(label)
            y_score.append(float(score))
            per_file.append({"file": path.name, "label": label, "score": round(float(score), 4)})
        except (ImageValidationError, ModelLoadError, InferenceError) as exc:
            errors.append({"file": path.name, "error": f"{type(exc).__name__}: {exc}"})
            print(f"  ! {path.name}: {type(exc).__name__}")
        if (i + 1) % 25 == 0:
            print(f"  ... {i + 1}/{len(dataset)} processed")
    elapsed = time.perf_counter() - start

    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        precision_recall_fscore_support,
        roc_auc_score,
    )

    report: dict = {
        "detector": args.detector,
        "models": {
            "ai-image": config.AI_IMAGE_MODEL_ID if args.detector == "ai-image" else None,
            "deepfake": config.DEEPFAKE_MODEL_ID if args.detector == "deepfake" else None,
        },
        "dataset": {"real": len(real_files), "fake": len(fake_files),
                    "real_dir": str(args.real_dir), "fake_dir": str(args.fake_dir)},
        "decision_boundary": config.DECISION_BOUNDARY,
        "n_scored": len(y_true),
        "n_errors": len(errors),
        "n_skipped_no_face": skipped,
        "per_file_predictions": per_file,
        "elapsed_s": round(elapsed, 1),
        "errors": errors,
    }

    if len(y_true) < 2 or len(set(y_true)) < 2:
        report["metrics"] = None
        report["metrics_note"] = (
            "Not enough scored images from BOTH classes to compute metrics - "
            "provide more labelled data. Nothing was fabricated."
        )
    else:
        y_pred = [1 if s >= config.DECISION_BOUNDARY else 0 for s in y_score]
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        report["metrics"] = {
            "accuracy": round(accuracy_score(y_true, y_pred), 4),
            "precision_fake": round(float(precision), 4),
            "recall_fake": round(float(recall), 4),
            "f1_fake": round(float(f1), 4),
            "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
            "roc_auc": None,
        }
        try:
            report["metrics"]["roc_auc"] = round(float(roc_auc_score(y_true, y_score)), 4)
        except ValueError:
            pass

    print("\n================ MEASURED RESULTS ================")
    print(f"scored={report['n_scored']}  errors={report['n_errors']}  "
          f"skipped_no_face={report['n_skipped_no_face']}  elapsed={report['elapsed_s']}s")
    m = report["metrics"]
    if m:
        print(f"accuracy      : {m['accuracy']}")
        print(f"precision(fake): {m['precision_fake']}")
        print(f"recall   (fake): {m['recall_fake']}")
        print(f"f1       (fake): {m['f1_fake']}")
        print(f"confusion     : {m['confusion_matrix']}")
        print(f"roc_auc       : {m['roc_auc']}")
    else:
        print(report["metrics_note"])

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2))
        print(f"\nJSON report written to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
