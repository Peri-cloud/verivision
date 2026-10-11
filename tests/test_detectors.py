"""Detector-level tests with MOCKED checkpoints (no network, no weights).

These mocks are used only in tests - the application never substitutes a
dummy predictor. Coverage: missing weights, network download failure,
inference exceptions, label-mapping validation, accelerator fallback.
"""
from __future__ import annotations

import pytest
import torch

import config
from models import common as model_common
from models.ai_image_detector import AIImageDetector
from models.deepfake_detector import DeepfakeDetector


# ------------------------------------------------------------- stub model
class StubLogits:
    def __init__(self, tensor):
        self.logits = tensor


class StubModel:
    """Minimal stand-in for a HF image-classification model."""

    def __init__(self, id2label, fail_with: Exception | None = None):
        self.config = type("Cfg", (), {"id2label": id2label})()
        self.fail_with = fail_with
        self.to_calls: list[str] = []

    def to(self, device):
        self.to_calls.append(str(device))
        return self

    def eval(self):
        return self

    def __call__(self, **kwargs):
        if self.fail_with is not None:
            raise self.fail_with
        return StubLogits(torch.tensor([[-2.0, 2.0]]))


class StubProcessor:
    def __call__(self, images=None, return_tensors=None):
        return {"pixel_values": torch.zeros(1, 3, 8, 8)}


class DeviceAgnosticTensor:
    """Simulates a tensor that can always move to the accelerator, so the
    simulated failure happens inside the MODEL forward pass (as on a real
    Mac with MPS) regardless of the host OS."""

    def to(self, device):
        return self


class AgnosticProcessor:
    def __call__(self, images=None, return_tensors=None):
        return {"pixel_values": DeviceAgnosticTensor()}


def patch_loaders(monkeypatch, model: StubModel, processor: StubProcessor | None = None):
    monkeypatch.setattr(
        model_common,
        "AutoImageProcessor",
        type("P", (), {"from_pretrained": staticmethod(lambda *_a, **_k: processor or StubProcessor())}),
    )
    monkeypatch.setattr(
        model_common,
        "AutoModelForImageClassification",
        type("M", (), {"from_pretrained": staticmethod(lambda *_a, **_k: model)}),
    )


class TestMissingWeights:
    def test_missing_weights_raise_model_load_error(self, monkeypatch):
        def boom(*_a, **_k):
            raise OSError(
                "haywoodsloan/ai-image-detector-deploy does not appear to have a "
                "file named model.safetensors"
            )

        monkeypatch.setattr(
            model_common, "AutoImageProcessor",
            type("P", (), {"from_pretrained": staticmethod(boom)}),
        )
        with pytest.raises(model_common.ModelLoadError, match="download_models"):
            AIImageDetector().load()


class TestNetworkFailure:
    @pytest.mark.parametrize(
        "message",
        [
            "HTTPConnectionPool(host='huggingface.co', port=443): Max retries exceeded "
            "with url: /api/models/x (Caused by NameResolutionError)",
            "Connection to huggingface.co timed out. (connect timeout=10)",
        ],
    )
    def test_network_error_gives_offline_guidance(self, monkeypatch, message):
        def boom(*_a, **_k):
            raise OSError(message)

        monkeypatch.setattr(
            model_common, "AutoModelForImageClassification",
            type("M", (), {"from_pretrained": staticmethod(boom)}),
        )
        with pytest.raises(model_common.ModelLoadError, match="internet connection"):
            DeepfakeDetector().load()


class TestLabelMapping:
    def test_wrong_labels_fail_loudly(self, monkeypatch):
        patch_loaders(monkeypatch, StubModel({0: "cat", 1: "dog"}))
        with pytest.raises(model_common.LabelMappingError, match="label"):
            AIImageDetector().load()

    def test_missing_id2label_fails(self, monkeypatch):
        model = StubModel(None)
        model.config.id2label = None
        patch_loaders(monkeypatch, model)
        with pytest.raises(model_common.LabelMappingError):
            AIImageDetector().load()

    def test_correct_mapping_loads_and_predicts(self, monkeypatch):
        patch_loaders(
            monkeypatch,
            StubModel({0: "artificial", 1: "real"}),
        )
        det = AIImageDetector().load()
        assert det.id2label == {0: "artificial", 1: "real"}
        from PIL import Image

        raw = det.predict(Image.new("RGB", (64, 64)))
        # stub logits [-2, 2] -> softmax ~ [0.018, 0.982]; fake class index 0
        assert raw.probabilities["artificial"] == pytest.approx(0.018, abs=1e-3)
        assert raw.fake_score == pytest.approx(0.018, abs=1e-3)
        assert raw.device == "cpu"


class TestSigmoidOutputMode:
    """Model B (CommunityForensics) uses a single sigmoid logit."""

    class SingleLogitModel(StubModel):
        def __init__(self):
            super().__init__({0: "LABEL_0"})

        def __call__(self, **kwargs):
            return StubLogits(torch.tensor([[-3.0]]))  # sigmoid ~ 0.047

    def test_sigmoid_semantics(self, monkeypatch):
        model = self.SingleLogitModel()
        patch_loaders(monkeypatch, model, processor=AgnosticProcessor())
        det = DeepfakeDetector().load()
        from PIL import Image

        raw = det.predict(Image.new("RGB", (64, 64)))
        assert raw.fake_score == pytest.approx(0.0474, abs=1e-3)
        assert raw.probabilities["fake"] == pytest.approx(0.0474, abs=1e-3)
        assert raw.probabilities["real"] == pytest.approx(0.9526, abs=1e-3)

    def test_sigmoid_rejects_multi_class_checkpoint(self, monkeypatch):
        # id2label has 2 entries -> structure mismatch for sigmoid mode
        patch_loaders(monkeypatch, StubModel({0: "Real", 1: "Fake"}))
        with pytest.raises(model_common.LabelMappingError, match="sigmoid"):
            DeepfakeDetector().load()

    def test_sigmoid_rejects_wrong_logit_count_at_runtime(self, monkeypatch):
        """A checkpoint that returns two logits must fail loudly in sigmoid mode
        instead of silently picking one."""
        patch_loaders(monkeypatch, self.SingleLogitModel(), processor=AgnosticProcessor())
        det = DeepfakeDetector().load()
        assert det.output_mode == "sigmoid"
        from models.common import run_classifier

        class TwoLogit:
            config = type("Cfg", (), {"id2label": {0: "LABEL_0"}})()

            def to(self, device):
                return self

            def eval(self):
                return self

            def __call__(self, **kwargs):
                return StubLogits(torch.tensor([[0.5, 0.5]]))

        from PIL import Image

        with pytest.raises(model_common.InferenceError, match="one logit"):
            run_classifier(
                AgnosticProcessor(), TwoLogit(), "cpu", Image.new("RGB", (32, 32)),
                fake_class_index=0, model_id="mock", output_mode="sigmoid",
            )


class TestInferenceExceptions:
    class FlakyMPSModel(StubModel):
        """Raises on the first call made while on 'mps', succeeds afterwards.
        Overrides __call__ at CLASS level so the dunder is honoured."""

        def __init__(self, id2label):
            super().__init__(id2label)
            self.n_calls = 0

        def __call__(self, **kwargs):
            self.n_calls += 1
            if self.n_calls == 1 and self.to_calls and self.to_calls[-1] == "mps":
                raise RuntimeError("MPS op not supported")
            return StubLogits(torch.tensor([[-2.0, 2.0]]))

    def test_mps_failure_falls_back_to_cpu(self, monkeypatch):
        """First forward pass raises (simulated accelerator op failure); after
        the detector switches to CPU the same input succeeds."""
        model = self.FlakyMPSModel({0: "artificial", 1: "real"})
        patch_loaders(monkeypatch, model, processor=AgnosticProcessor())
        det = AIImageDetector(device="mps")
        det.load()
        from PIL import Image

        raw = det.predict(Image.new("RGB", (64, 64)))
        assert det.device == "cpu"
        assert any("retried on CPU" in w for w in raw.warnings)
        assert model.n_calls == 2
        assert raw.device == "cpu"

    def test_inference_error_propagates_on_cpu(self, monkeypatch):
        patch_loaders(
            monkeypatch,
            StubModel({0: "artificial", 1: "real"}, fail_with=RuntimeError("boom")),
        )
        det = AIImageDetector().load()
        from PIL import Image

        with pytest.raises(model_common.InferenceError, match="boom"):
            det.predict(Image.new("RGB", (64, 64)))


class TestBothDetectorIndependence:
    def test_run_full_analysis_returns_both_results(self, monkeypatch):
        """Both-detector mode must keep the two outputs separate."""
        from analysis import run_full_analysis
        from conftest import FakeAIDetector, FakeDeepfakeDetector, FakeFaceDetector, make_image
        from models.face_detector import FaceBox

        patch_loaders(monkeypatch, StubModel({0: "artificial", 1: "real"}))
        ai = FakeAIDetector(fake_score=0.9)
        faces = FakeFaceDetector([FaceBox(10, 10, 120, 120, 0.9, "mock")])
        deep = FakeDeepfakeDetector(0.1)

        image = make_image(600, 400)  # real PIL image: crop_face touches pixels
        out = run_full_analysis(
            image,
            ai_detector=ai,
            face_detector=faces,
            deepfake_detector=deep,
            mode="Both Detectors",
        )
        assert out["ai_image"] is not None and out["deepfake"] is not None
        assert out["ai_image"]["verdict"].value != out["deepfake"]["summary"]["verdict"].value
        assert out["ai_image"]["fake_score"] == 0.9
        assert out["deepfake"]["faces"][0]["fake_score"] == 0.1
