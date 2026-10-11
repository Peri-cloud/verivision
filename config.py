"""Central configuration for VeriVision AI.

Every tunable constant lives here. Model identifiers, label mappings and the
decision rule below were verified empirically during development - see the
README section "Model selection rationale" for the evidence and limitations.
"""
from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------- paths ----
BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "assets"
SAMPLE_IMAGES_DIR = ASSETS_DIR / "sample_images"

# Where model weights / caches are stored (this directory is git-ignored).
# Override with the VERIVISION_MODEL_CACHE environment variable, e.g.:
#   export VERIVISION_MODEL_CACHE=/path/to/big_disk/verivision_cache
WEIGHTS_DIR = Path(os.environ.get("VERIVISION_MODEL_CACHE", BASE_DIR / ".model_cache"))
HF_CACHE_DIR = WEIGHTS_DIR / "hf_hub"
YUNET_WEIGHTS_PATH = WEIGHTS_DIR / "yunet" / "face_detection_yunet_2023mar.onnx"

# Route the Hugging Face download cache into the project-local weights dir
# (respects an HF_HOME the user may already have set).
os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

# ------------------------------------------------------------- uploads ----
ALLOWED_FORMATS = {"JPEG", "PNG"}          # verified from the file's real bytes
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
MAX_FILE_BYTES = 25 * 1024 * 1024          # reject uploads above 25 MB
MAX_PIXELS = 50_000_000                    # reject images above 50 megapixels
MIN_DIMENSION_PX = 32                      # reject images smaller than 32 px

# -------------------------------------------------------------- models ----
# Model A - AI-generated image detector.
# Verified: SwinV2, 256x256 input, id2label = {0: 'artificial', 1: 'real'},
# Apache-2.0. Empirically 5/5 correct on the bundled dev smoke samples.
AI_IMAGE_MODEL_ID = os.environ.get(
    "VERIVISION_AI_IMAGE_MODEL", "haywoodsloan/ai-image-detector-deploy"
)
AI_IMAGE_OUTPUT_MODE = "softmax"           # two-class softmax head
AI_IMAGE_FAKE_CLASS_INDEX = 0              # index of the "artificial" class
AI_IMAGE_EXPECTED_LABELS = {"artificial", "real"}
AI_IMAGE_SIZE_PX = 256

# Model B - deepfake face detector (facial manipulation / synthetic-face forensics).
# Verified: ViT-S/16, 384x384 input (shortest-edge 440 + centre crop), MIT.
# Trained on 2.7M samples across 4,803 generators (Community Forensics,
# arXiv:2411.04125). Output = SINGLE sigmoid logit where 1.0 = fake.
# id2label is {0: 'LABEL_0'}; semantics come from the official model card,
# and common.py validates num_labels == 1 at load time.
DEEPFAKE_MODEL_ID = os.environ.get(
    "VERIVISION_DEEPFAKE_MODEL", "buildborderless/CommunityForensics-DeepfakeDet-ViT"
)
DEEPFAKE_OUTPUT_MODE = "sigmoid"           # 'sigmoid' (1 logit) | 'softmax' (2 classes)
DEEPFAKE_FAKE_CLASS_INDEX = 0              # only used when output_mode = 'softmax'
DEEPFAKE_EXPECTED_LABELS = {"fake", "real"}
DEEPFAKE_SIZE_PX = 384

# YuNet face detector (Apache-2.0, from the official opencv_zoo).
YUNET_DOWNLOAD_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)

# --------------------------------------------------------- decision rule --
# DECISION_BOUNDARY = 0.5 is simply the argmax boundary of a two-class
# softmax output. It is NOT a calibrated probability threshold: these
# checkpoints ship without calibration data. See README, "How inference
# works".
DECISION_BOUNDARY = 0.5
# Scores within +/- UNCERTAINTY_BAND of the boundary are reported as
# "Inconclusive". This band is a deliberately conservative *interface
# heuristic* chosen so borderline images are surfaced to the user instead of
# being forced into a binary verdict. It is NOT a validated calibration
# procedure (see README).
UNCERTAINTY_BAND = 0.10

# --------------------------------------------------------------- faces ----
DETECT_MAX_SIDE = 1024      # faces are detected on a downscaled copy, then
                            # boxes are scaled back (much fewer false hits)
MIN_FACE_SIZE_PX = 64       # faces smaller than this are not analysed
FACE_CROP_MARGIN = 0.25     # 25% margin around each detected face box
YUNET_SCORE_THRESHOLD = 0.6
HAAR_SCALE_FACTOR = 1.1
HAAR_MIN_NEIGHBOURS = 5
MAX_FACES = 8               # analyse at most this many faces per image

# -------------------------------------------------------------- app UI ----
APP_TITLE = "VeriVision AI"
APP_SUBTITLE = (
    "Real image, AI-generated image & deepfake face detection "
    "with transparent, appropriately qualified results"
)
DISCLAIMER = (
    "Predictions are probabilistic model outputs and are NOT proof of "
    "authenticity. Scores are raw, uncalibrated model outputs. Do not use "
    "these results to accuse, defame or judge any person or organisation."
)
