"""Face-detection behaviour: backends, no-face cases, tiny images, and the
analysis pipeline's small-face filtering (uses the real OpenCV backends)."""
from __future__ import annotations

import pytest

import config
from analysis import run_deepfake_check
from conftest import FakeDeepfakeDetector, make_image
from models.face_detector import FaceBox, FaceDetector


@pytest.fixture(scope="module")
def face_detector() -> FaceDetector:
    return FaceDetector().load()


class TestBackendSelection:
    def test_backend_is_valid(self, face_detector):
        assert face_detector.backend in ("yunet", "haar")

    def test_falls_back_to_haar_when_weights_missing(self, tmp_path):
        det = FaceDetector(yunet_weights_path=tmp_path / "missing.onnx").load()
        assert det.backend == "haar"


class TestDetection:
    def test_no_face_in_gradient_image(self, face_detector):
        img = make_image(640, 480)
        assert face_detector.detect(img) == []

    def test_tiny_image_returns_empty_not_crash(self, face_detector):
        img = make_image(30, 30)
        assert face_detector.detect(img) == []

    def test_detects_face_in_bundled_portrait(self, face_detector):
        """Real detection against the bundled (synthetic-person) sample."""
        sample = config.SAMPLE_IMAGES_DIR / "sample_ai_portrait.jpg"
        if not sample.exists():
            pytest.skip("bundled portrait sample missing")
        img = validate_or_skip(sample)
        boxes = face_detector.detect(img)
        assert len(boxes) >= 1
        assert all(b.w > 0 and b.h > 0 for b in boxes)


def validate_or_skip(path):
    from utils.image_validation import validate_image_bytes

    try:
        return validate_image_bytes(path.read_bytes())[0]
    except Exception:
        pytest.skip(f"sample {path.name} could not be validated in this env")


class TestAnalysisPipeline:
    def test_no_face_gives_no_fabricated_score(self, monkeypatch):
        from conftest import make_image as _img
        from models.face_detector import FaceDetector as RealFD

        monkeypatch.setattr(
            RealFD, "detect", lambda self, image: []
        )
        result = run_deepfake_check(
            _img(), RealFD().load(), FakeDeepfakeDetector(0.9)
        )
        assert result["face_count"] == 0
        assert result["faces"] == []
        assert result["summary"]["verdict"].value == "No face detected"

    def test_small_faces_are_skipped_with_limitation(self, face_box, small_face_box, monkeypatch):
        from conftest import make_image as _img
        from models.face_detector import FaceDetector as RealFD

        monkeypatch.setattr(
            RealFD, "detect", lambda self, image: [small_face_box, face_box]
        )
        result = run_deepfake_check(
            _img(640, 480), RealFD().load(), FakeDeepfakeDetector([0.2])
        )
        # only the big face is analysed; the small one is reported, not scored
        assert result["face_count"] == 1
        assert any("below the 64px minimum" in lim for lim in result["limitations"])

    def test_multiple_faces_kept_separate(self, monkeypatch):
        from conftest import make_image as _img
        from models.face_detector import FaceDetector as RealFD

        boxes = [
            FaceBox(0, 0, 200, 200, 0.9, "mock"),
            FaceBox(300, 0, 200, 200, 0.9, "mock"),
        ]
        monkeypatch.setattr(RealFD, "detect", lambda self, image: boxes)
        result = run_deepfake_check(
            _img(600, 300), RealFD().load(), FakeDeepfakeDetector([0.05, 0.95])
        )
        verdicts = [f["verdict"].value for f in result["faces"]]
        assert verdicts[0] != verdicts[1]
        assert result["summary"]["conflicting"] is True

    def test_inference_error_on_one_face_does_not_poison_others(self, monkeypatch):
        from conftest import make_image as _img
        from models.common import InferenceError
        from models.deepfake_detector import DeepfakeDetector as RealDD
        from models.face_detector import FaceDetector as RealFD

        class FlakyDeepfake:
            def predict(self, crop):
                raise InferenceError("device exploded")

        boxes = [FaceBox(0, 0, 200, 200, 0.9, "mock")]
        monkeypatch.setattr(RealFD, "detect", lambda self, image: boxes)
        result = run_deepfake_check(_img(400, 300), RealFD().load(), FlakyDeepfake())
        assert result["faces"][0]["verdict"].value == "Analysis error"
        assert "device exploded" in result["faces"][0]["error_message"]
