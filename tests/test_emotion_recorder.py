"""
Tests for pipeline.emotion_recorder.EmotionRecorder.

All tests use mock objects for webcam, face detector, and classifier.
No hardware dependencies.

Key Phase 0 interfaces this file mocks:
- WebcamCapture.read_frame() -> (timestamp_ms: float, frame: np.ndarray)
  Raises RuntimeError on failure.
- WebcamCapture.is_open -> bool property
- WebcamCapture.open() / release()
- FaceDetector.detect(frame) -> (x,y,w,h) tuple or None  (NOT a list)
- FaceDetector.crop_face(frame) -> np.ndarray or None
- EmotionClassifier.classify(face_roi) -> (emotion, confidence, all_scores)
"""

import csv
import sys
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.emotion_recorder import EmotionRecorder, EmotionRecord, CSV_COLUMNS
from utils.timer import SessionTimer


# ---------------------------------------------------------------------------
# Fixtures — mock all hardware dependencies
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_frame():
    """A 480x640 BGR numpy array simulating a webcam frame."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def mock_webcam(fake_frame):
    """Webcam mock: is_open=True, read_frame returns (0.0, frame)."""
    webcam = MagicMock()
    type(webcam).is_open = PropertyMock(return_value=True)
    webcam.read_frame.return_value = (0.0, fake_frame)
    return webcam


@pytest.fixture
def mock_webcam_read_fails():
    """Webcam mock: read_frame raises RuntimeError (simulates dropped frame)."""
    webcam = MagicMock()
    type(webcam).is_open = PropertyMock(return_value=True)
    webcam.read_frame.side_effect = RuntimeError("Failed to read frame.")
    return webcam


@pytest.fixture
def mock_face_detector(fake_frame):
    """Face detector mock: detect returns a single bbox, crop_face returns a region."""
    detector = MagicMock()
    detector.detect.return_value = (100, 100, 150, 150)
    detector.crop_face.return_value = fake_frame[100:250, 100:250]
    return detector


@pytest.fixture
def mock_face_detector_no_face():
    """Face detector mock: no face found."""
    detector = MagicMock()
    detector.detect.return_value = None
    detector.crop_face.return_value = None
    return detector


@pytest.fixture
def mock_classifier():
    """Classifier mock: returns happy with full score dict."""
    classifier = MagicMock()
    classifier.classify.return_value = (
        "happy",
        0.87,
        {
            "angry": 0.01,
            "disgusted": 0.01,
            "fearful": 0.01,
            "happy": 0.87,
            "sad": 0.02,
            "surprised": 0.05,
            "neutral": 0.03,
        },
    )
    return classifier


@pytest.fixture
def recorder(mock_webcam, mock_face_detector, mock_classifier):
    return EmotionRecorder(
        webcam=mock_webcam,
        face_detector=mock_face_detector,
        emotion_classifier=mock_classifier,
    )


# ---------------------------------------------------------------------------
# Session lifecycle tests
# ---------------------------------------------------------------------------

class TestRecorderLifecycle:

    def test_record_frame_before_start_raises(self, recorder):
        with pytest.raises(RuntimeError, match="before calling start"):
            recorder.record_frame()

    def test_start_clears_previous_records(self, recorder):
        recorder.start()
        recorder.record_frame()
        assert len(recorder.records) == 1

        # Restart should clear
        recorder.start()
        assert len(recorder.records) == 0

    def test_stop_freezes_timer(self, recorder):
        recorder.start()
        recorder.record_frame()
        recorder.stop()
        assert recorder.timer.is_stopped

    def test_start_opens_webcam_if_closed(self, mock_face_detector, mock_classifier):
        webcam = MagicMock()
        type(webcam).is_open = PropertyMock(return_value=False)
        recorder = EmotionRecorder(
            webcam=webcam,
            face_detector=mock_face_detector,
            emotion_classifier=mock_classifier,
        )
        recorder.start()
        webcam.open.assert_called_once()

    def test_stop_releases_webcam(self, recorder):
        recorder.start()
        recorder.stop()
        recorder._webcam.release.assert_called_once()


# ---------------------------------------------------------------------------
# Frame recording tests
# ---------------------------------------------------------------------------

class TestRecordFrame:

    def test_record_frame_returns_emotion_record(self, recorder):
        recorder.start()
        record = recorder.record_frame()
        assert isinstance(record, EmotionRecord)
        assert record.emotion == "happy"
        assert record.confidence == 0.87
        assert record.face_detected is True

    def test_record_frame_appends_to_records(self, recorder):
        recorder.start()
        recorder.record_frame()
        recorder.record_frame()
        recorder.record_frame()
        assert len(recorder.records) == 3

    def test_timestamps_are_non_decreasing(self, recorder):
        recorder.start()
        records = [recorder.record_frame() for _ in range(5)]
        timestamps = [r.timestamp_ms for r in records]
        assert timestamps == sorted(timestamps)

    def test_webcam_failure_returns_none(
        self, mock_webcam_read_fails, mock_face_detector, mock_classifier
    ):
        recorder = EmotionRecorder(
            webcam=mock_webcam_read_fails,
            face_detector=mock_face_detector,
            emotion_classifier=mock_classifier,
        )
        recorder.start()
        result = recorder.record_frame()
        assert result is None
        assert len(recorder.records) == 0

    def test_no_face_records_no_face_label(
        self, mock_webcam, mock_face_detector_no_face, mock_classifier
    ):
        recorder = EmotionRecorder(
            webcam=mock_webcam,
            face_detector=mock_face_detector_no_face,
            emotion_classifier=mock_classifier,
        )
        recorder.start()
        record = recorder.record_frame()
        assert record.emotion == "no_face"
        assert record.face_detected is False
        assert record.confidence == 0.0

    def test_classifier_receives_cropped_face(
        self, mock_webcam, mock_face_detector, mock_classifier
    ):
        """Verify the classifier gets the output of crop_face, not the raw frame."""
        recorder = EmotionRecorder(
            webcam=mock_webcam,
            face_detector=mock_face_detector,
            emotion_classifier=mock_classifier,
        )
        recorder.start()
        recorder.record_frame()
        # crop_face was called, and its return value was passed to classify
        mock_face_detector.crop_face.assert_called_once()
        mock_classifier.classify.assert_called_once()
        classify_arg = mock_classifier.classify.call_args[0][0]
        assert classify_arg.shape == (150, 150, 3)


# ---------------------------------------------------------------------------
# EmotionRecord tests
# ---------------------------------------------------------------------------

class TestEmotionRecord:

    def test_to_csv_row_has_all_columns(self):
        record = EmotionRecord(
            timestamp_ms=1000,
            emotion="happy",
            confidence=0.87,
            face_detected=True,
            all_scores={"happy": 0.87, "neutral": 0.13},
        )
        row = record.to_csv_row()
        for col in CSV_COLUMNS:
            assert col in row

    def test_to_csv_row_missing_scores_default_to_zero(self):
        record = EmotionRecord(
            timestamp_ms=500,
            emotion="neutral",
            confidence=0.5,
            face_detected=True,
            all_scores={"neutral": 0.5},
        )
        row = record.to_csv_row()
        assert row["happy"] == 0.0
        assert row["neutral"] == 0.5


# ---------------------------------------------------------------------------
# CSV export tests
# ---------------------------------------------------------------------------

class TestExportCSV:

    def test_export_creates_file(self, recorder, tmp_path):
        with patch("pipeline.emotion_recorder.RECORDER_OUTPUT_DIR", str(tmp_path)):
            recorder.start()
            recorder.record_frame()
            recorder.stop()
            path = recorder.export_csv("test_output.csv")
            assert path.exists()

    def test_export_empty_session_raises(self, recorder):
        with pytest.raises(ValueError, match="No records"):
            recorder.export_csv()

    def test_export_csv_has_correct_headers(self, recorder, tmp_path):
        with patch("pipeline.emotion_recorder.RECORDER_OUTPUT_DIR", str(tmp_path)):
            recorder.start()
            recorder.record_frame()
            recorder.stop()
            path = recorder.export_csv("test_headers.csv")

            with open(path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                assert list(reader.fieldnames) == CSV_COLUMNS

    def test_export_csv_row_count_matches_records(self, recorder, tmp_path):
        with patch("pipeline.emotion_recorder.RECORDER_OUTPUT_DIR", str(tmp_path)):
            recorder.start()
            for _ in range(5):
                recorder.record_frame()
            recorder.stop()
            path = recorder.export_csv("test_count.csv")

            with open(path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                assert len(rows) == 5

    def test_export_csv_default_filename(self, recorder, tmp_path):
        with patch("pipeline.emotion_recorder.RECORDER_OUTPUT_DIR", str(tmp_path)):
            recorder.start()
            recorder.record_frame()
            recorder.stop()
            path = recorder.export_csv()
            assert path.name == "emotion_session.csv"
