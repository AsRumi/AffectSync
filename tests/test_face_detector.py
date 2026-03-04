"""
Unit tests for pipeline.face_detector.FaceDetector.

All tests use synthetic numpy arrays — no webcam or real images required.

Note: cv2.CascadeClassifier is a C++ extension object whose methods
cannot be patched with mock.patch.object. Instead we:
  - Patch cv2.CascadeClassifier at the MODULE level for detect() tests
  - Patch FaceDetector.detect (pure Python) for crop_face() tests
"""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def blank_frame():
    """A 480x640 black BGR frame — no face present."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def random_frame():
    """A 480x640 random BGR frame for cropping tests."""
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


class TestFaceDetectorInit:
    def test_init_loads_cascade(self):
        from pipeline.face_detector import FaceDetector
        detector = FaceDetector()
        assert detector._classifier is not None


class TestFaceDetectorRealCascade:
    """Tests that use the real Haar cascade (no mocking needed)."""

    def test_detect_returns_none_on_blank_frame(self, blank_frame):
        from pipeline.face_detector import FaceDetector
        detector = FaceDetector()
        result = detector.detect(blank_frame)
        assert result is None

    def test_crop_face_returns_none_on_blank_frame(self, blank_frame):
        from pipeline.face_detector import FaceDetector
        detector = FaceDetector()
        result = detector.crop_face(blank_frame)
        assert result is None


class TestFaceDetectorMockedCascade:
    """Tests that need to simulate face detections by mocking the cascade."""

    def _build_detector_with_mock(self, mock_cascade_cls):
        """Reload the module so it picks up the mocked CascadeClassifier."""
        from importlib import reload
        import pipeline.face_detector as fd_module
        reload(fd_module)
        return fd_module.FaceDetector()

    @patch("pipeline.face_detector.cv2.CascadeClassifier")
    def test_detect_returns_tuple_when_face_found(self, mock_cascade_cls):
        fake_faces = np.array([[100, 100, 80, 80]])
        mock_cascade_cls.return_value.detectMultiScale.return_value = fake_faces

        detector = self._build_detector_with_mock(mock_cascade_cls)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        bbox = detector.detect(frame)

        assert bbox is not None
        assert len(bbox) == 4
        x, y, w, h = bbox
        assert x == 100 and y == 100 and w == 80 and h == 80

    @patch("pipeline.face_detector.cv2.CascadeClassifier")
    def test_detect_picks_largest_face(self, mock_cascade_cls):
        fake_faces = np.array([
            [50, 50, 30, 30],
            [200, 200, 100, 100],
        ])
        mock_cascade_cls.return_value.detectMultiScale.return_value = fake_faces

        detector = self._build_detector_with_mock(mock_cascade_cls)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        bbox = detector.detect(frame)

        assert bbox is not None
        x, y, w, h = bbox
        assert w == 100 and h == 100


class TestCropFaceLogic:
    """Test cropping logic by mocking FaceDetector.detect (pure Python, mockable)."""

    def test_crop_returns_correct_region(self, random_frame):
        from pipeline.face_detector import FaceDetector
        detector = FaceDetector()

        with patch.object(detector, "detect", return_value=(100, 100, 80, 80)):
            crop = detector.crop_face(random_frame, padding=0.0)
            assert crop is not None
            assert crop.shape[0] == 80
            assert crop.shape[1] == 80
            assert crop.shape[2] == 3

    def test_padding_expands_region(self, random_frame):
        from pipeline.face_detector import FaceDetector
        detector = FaceDetector()

        with patch.object(detector, "detect", return_value=(100, 100, 80, 80)):
            crop_no_pad = detector.crop_face(random_frame, padding=0.0)
            crop_with_pad = detector.crop_face(random_frame, padding=0.2)
            assert crop_with_pad.shape[0] >= crop_no_pad.shape[0]
            assert crop_with_pad.shape[1] >= crop_no_pad.shape[1]

    def test_padding_clamps_to_frame_boundary(self, random_frame):
        from pipeline.face_detector import FaceDetector
        detector = FaceDetector()

        with patch.object(detector, "detect", return_value=(0, 0, 80, 80)):
            crop = detector.crop_face(random_frame, padding=0.5)
            assert crop is not None
            assert crop.shape[0] > 0 and crop.shape[1] > 0
