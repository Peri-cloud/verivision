"""Shared model-loading infrastructure.

Provides device resolution (CUDA -> MPS -> CPU), guarded checkpoint loading
with friendly offline/missing-weights errors, strict label-mapping validation
so a checkpoint with unexpected semantics fails loudly instead of silently
producing nonsense, and the timed inference call used by every detector.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification

from utils.preprocessing import ensure_rgb


class ModelLoadError(RuntimeError):
    """A detector's weights could not be loaded (offline, missing, corrupt)."""


class LabelMappingError(ModelLoadError):
    """The checkpoint's label mapping does not match the documented mapping."""


class InferenceError(RuntimeError):
    """A loaded model failed while running inference."""


def resolve_device() -> str:
    """Pick the best available device: CUDA, then Apple MPS, then CPU."""
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class RawPrediction:
    """Raw, UI-agnostic detector output shared by both models."""

    probabilities: dict[str, float]
    fake_score: float
    model_id: str
    device: str
    inference_ms: float
    warnings: list[str] = field(default_factory=list)


def load_image_classifier(
    model_id: str,
    *,
    expected_labels: set[str],
    fake_class_index: int,
    device: str,
    output_mode: str = "softmax",
):
    """Load an image-classification checkpoint with strict validation.

    Two output modes are supported:
      * ``softmax`` - multi-class head; ``fake_class_index`` selects the
        fake class and probabilities come from softmax.
      * ``sigmoid`` - single-logit head; the sigmoid of the logit is the
        fake probability (used by CommunityForensics-DeepfakeDet-ViT, whose
        model card defines this semantics explicitly).

    Returns ``(processor, model, id2label)``. Raises:
      * ModelLoadError  - weights missing / network failure / unexpected error
      * LabelMappingError - the checkpoint's labels differ from the
        documented mapping in config.py (this is checked so a swapped
        checkpoint can never silently invert real/fake semantics)
    """
    if output_mode not in ("softmax", "sigmoid"):
        raise ModelLoadError(f"Unknown output mode '{output_mode}' in config.py")
    try:
        processor = AutoImageProcessor.from_pretrained(model_id)
        model = AutoModelForImageClassification.from_pretrained(model_id)
    except OSError as exc:
        msg = str(exc).lower()
        network_keys = (
            "connection", "timed out", "timeout", "network", "getaddrinfo",
            "name or service", "temporary failure", "max retries",
            "connectionerror", "proxy", "certificat",
        )
        if any(k in msg for k in network_keys):
            raise ModelLoadError(
                f"Could not download model '{model_id}' from the Hugging Face Hub "
                "(network problem). Check your internet connection, then either:\n"
                "  1. re-run this app, or\n"
                "  2. pre-download the weights once:  python scripts/download_models.py\n"
                "If huggingface.co is blocked on your network, set the mirror:  "
                "set HF_ENDPOINT=https://hf-mirror.com   (Windows)  |  "
                "export HF_ENDPOINT=https://hf-mirror.com   (macOS/Linux)"
            ) from exc
        raise ModelLoadError(
            f"Weights for '{model_id}' are not available locally and could not be "
            "loaded. Run  python scripts/download_models.py  to download them, or "
            "re-run this app while connected to the internet (weights are cached "
            "automatically after the first download)."
        ) from exc
    except Exception as exc:  # noqa: BLE001 - corrupt cache, OOM during load, etc.
        raise ModelLoadError(
            f"Unexpected failure while loading '{model_id}': {type(exc).__name__}: {exc}"
        ) from exc

    model.to(device)
    model.eval()

    raw_id2label = model.config.id2label
    if not isinstance(raw_id2label, dict) or not raw_id2label:
        raise LabelMappingError(
            f"Checkpoint '{model_id}' has no usable id2label mapping. Refusing to "
            "guess the real/fake semantics - update config.py with the verified "
            "mapping from the model card."
        )
    id2label = {int(k): str(v) for k, v in raw_id2label.items()}

    if output_mode == "sigmoid":
        # Single-class sigmoid head (e.g. CommunityForensics): semantics are
        # defined by the model card, so we validate the STRUCTURE instead of
        # class names - exactly one output logit.
        if len(id2label) != 1:
            raise LabelMappingError(
                f"Checkpoint '{model_id}' was configured with output_mode='sigmoid' "
                f"but exposes {len(id2label)} classes; refusing to guess semantics."
            )
    else:
        present = {v.strip().lower() for v in id2label.values()}
        missing = {lbl.lower() for lbl in expected_labels} - present
        if missing:
            raise LabelMappingError(
                f"Checkpoint '{model_id}' labels {sorted(present)} do not match the "
                f"documented mapping {sorted(expected_labels)} (missing {sorted(missing)}). "
                "Refusing to run with unknown label semantics."
            )
        if fake_class_index not in id2label:
            raise LabelMappingError(
                f"Checkpoint '{model_id}' has no class index {fake_class_index}; cannot "
                "identify the fake-class output. Update config.py."
            )

    return processor, model, id2label


def run_classifier(
    processor,
    model,
    device: str,
    image,
    *,
    fake_class_index: int,
    model_id: str,
    output_mode: str = "softmax",
    extra_warnings: list[str] | None = None,
) -> RawPrediction:
    """One timed forward pass. The returned fake_score is the model's fake
    probability under the configured output mode - an uncalibrated model
    output, never a calibrated probability of ground truth."""
    start = time.perf_counter()
    try:
        inputs = processor(images=ensure_rgb(image), return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.inference_mode():
            logits = model(**inputs).logits
        logits = logits.float()[0]
        if output_mode == "sigmoid":
            if logits.numel() != 1:
                raise InferenceError(
                    "Sigmoid output mode expects exactly one logit, got "
                    f"{logits.numel()} - check config.py."
                )
            fake_score = float(torch.sigmoid(logits[0]))
            probabilities = {"fake": fake_score, "real": 1.0 - fake_score}
        else:
            probs = torch.softmax(logits, dim=-1).detach().cpu()
            id2label = {int(k): str(v) for k, v in model.config.id2label.items()}
            probabilities = {id2label[i]: float(p) for i, p in enumerate(probs)}
            fake_score = float(probs[fake_class_index])
    except InferenceError:
        raise
    except RuntimeError as exc:
        raise InferenceError(
            f"Model inference failed on device '{device}': {exc}"
        ) from exc
    except Exception as exc:  # noqa: BLE001
        raise InferenceError(
            f"Model inference failed: {type(exc).__name__}: {exc}"
        ) from exc

    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return RawPrediction(
        probabilities=probabilities,
        fake_score=fake_score,
        model_id=model_id,
        device=device,
        inference_ms=elapsed_ms,
        warnings=list(extra_warnings or []),
    )
