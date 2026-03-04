"""
Unit tests for pipeline.webcam_capture.WebcamCapture.

All cv2.VideoCapture calls are mocked — no webcam hardware required.
"""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.webcam_capture import WebcamCapture


def _make_mock_capture(is_open: bool = True, read_success: bool = True):
    """Create a mock cv2.VideoCapture object."""
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = is_open
    if read_success:
        fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cap.read.return_value = (True, fake_frame)
    else:
        mock_cap.read.return_value = (False, None)
    mock_cap.get.return_value = 640.0  # For resolution logging
    mock_cap.set.return_value = True
    return mock_cap


class TestWebcamCapture:
    @patch("pipeline.webcam_capture.cv2.VideoCapture")
    def test_open_succeeds(self, mock_vc_class):
        mock_vc_class.return_value = _make_mock_capture(is_open=True)
        cam = WebcamCapture()
        cam.open()
        assert cam.is_open
        cam.release()

    @patch("pipeline.webcam_capture.cv2.VideoCapture")
    def test_open_raises_on_failure(self, mock_vc_class):
        mock_vc_class.return_value = _make_mock_capture(is_open=False)
        cam = WebcamCapture()
        with pytest.raises(RuntimeError, match="Cannot open webcam"):
            cam.open()

    @patch("pipeline.webcam_capture.cv2.VideoCapture")
    def test_context_manager(self, mock_vc_class):
        mock_cap = _make_mock_capture(is_open=True)
        mock_vc_class.return_value = mock_cap
        with WebcamCapture() as cam:
            assert cam.is_open
        mock_cap.release.assert_called_once()

    @patch("pipeline.webcam_capture.cv2.VideoCapture")
    def test_read_frame_returns_timestamp_and_array(self, mock_vc_class):
        mock_vc_class.return_value = _make_mock_capture()
        with WebcamCapture() as cam:
            ts, frame = cam.read_frame()
            assert isinstance(ts, float)
            assert ts >= 0
            assert isinstance(frame, np.ndarray)
            assert frame.shape == (480, 640, 3)

    @patch("pipeline.webcam_capture.cv2.VideoCapture")
    def test_read_frame_raises_when_closed(self, mock_vc_class):
        mock_vc_class.return_value = _make_mock_capture(is_open=False)
        cam = WebcamCapture()
        cam._cap = None
        with pytest.raises(RuntimeError, match="not open"):
            cam.read_frame()

    @patch("pipeline.webcam_capture.cv2.VideoCapture")
    def test_frames_generator_yields(self, mock_vc_class):
        mock_cap = _make_mock_capture()
        # After 3 reads, close the capture to stop the generator
        call_count = 0

        def side_effect_read():
            nonlocal call_count
            call_count += 1
            if call_count > 3:
                mock_cap.isOpened.return_value = False
                return (False, None)
            return (True, np.zeros((480, 640, 3), dtype=np.uint8))

        mock_cap.read.side_effect = side_effect_read
        mock_vc_class.return_value = mock_cap

        cam = WebcamCapture()
        cam.open()
        frames = list(cam.frames())
        assert len(frames) == 3
        cam.release()
