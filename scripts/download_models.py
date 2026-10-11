"""Pre-download all model weights so the app can run offline afterwards.

Usage:
    python scripts/download_models.py            # everything
    python scripts/download_models.py --ai-image # Model A only
    python scripts/download_models.py --deepfake # Model B only
    python scripts/download_models.py --yunet    # YuNet face detector only

Weights are stored in the project-local cache directory (config.WEIGHTS_DIR,
git-ignored) unless VERIVISION_MODEL_CACHE or HF_HOME is overridden.
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config


def download_hf_model(model_id: str, tag: str) -> bool:
    print(f"\n[{tag}] downloading {model_id} ...")
    try:
        from transformers import AutoImageProcessor, AutoModelForImageClassification

        proc = AutoImageProcessor.from_pretrained(model_id)
        model = AutoModelForImageClassification.from_pretrained(model_id)
        params_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6
        print(f"[{tag}] OK - {params_mb:.0f} MB of parameters, labels {model.config.id2label}")
        del model, proc
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[{tag}] FAILED: {type(exc).__name__}: {exc}")
        print(
            f"[{tag}] Hint: if huggingface.co is unreachable, set the mirror:\n"
            f"       export HF_ENDPOINT=https://hf-mirror.com   (macOS/Linux)\n"
            f"       set HF_ENDPOINT=https://hf-mirror.com    (Windows PowerShell: $env:HF_ENDPOINT=...)"
        )
        return False


def download_yunet() -> bool:
    dest = config.YUNET_WEIGHTS_PATH
    if dest.exists():
        print(f"\n[yunet] already present: {dest}")
        return True
    print(f"\n[yunet] downloading {config.YUNET_DOWNLOAD_URL}")
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(config.YUNET_DOWNLOAD_URL, dest)
        print(f"[yunet] OK - stored at {dest}")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[yunet] FAILED: {type(exc).__name__}: {exc}")
        print("[yunet] The app will fall back to OpenCV's bundled Haar cascade "
              "(works offline, slightly less accurate).")
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ai-image", action="store_true", help="Model A only")
    parser.add_argument("--deepfake", action="store_true", help="Model B only")
    parser.add_argument("--yunet", action="store_true", help="YuNet face detector only")
    args = parser.parse_args()

    everything = not (args.ai_image or args.deepfake or args.yunet)
    results = []

    if everything or args.yunet:
        results.append(("yunet", download_yunet()))
    if everything or args.ai_image:
        results.append(("ai-image", download_hf_model(config.AI_IMAGE_MODEL_ID, "ai-image")))
    if everything or args.deepfake:
        results.append(("deepfake", download_hf_model(config.DEEPFAKE_MODEL_ID, "deepfake")))

    print(f"\nWeights directory: {config.WEIGHTS_DIR}")
    ok = all(ok for _, ok in results)
    print("Summary:", ", ".join(f"{name}={'OK' if ok_ else 'FAILED'}" for name, ok_ in results))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
