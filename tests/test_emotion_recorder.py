"""Tests for pipeline.emotion_recorder.EmotionRecorder."""

import csv
import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

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
    webcam = MagicMock()
    webcam.read_frame.return_value = fake_frame
    return webcam


@pytest.fixture
def mock_webcam_no_frame():
    webcam = MagicMock()
    webcam.read_frame.return_value = None
    return webcam


@pytest.fixture
def mock_face_detector():
    detector = MagicMock()
    # Returns one face bounding box: (x, y, w, h)
    detector.detect.return_value = [(100, 100, 150, 150)]
    return detector


@pytest.fixture
def mock_face_detector_no_face():
    detector = MagicMock()
    detector.detect.return_value = []
    return detector


@pytest.fixture
def mock_classifier():
    classifier = MagicMock()
    classifier.classify.return_value = (
        "happy",
        0.87,
        {
            "happy": 0.87,
            "sad": 0.02,
            "angry": 0.01,
            "surprised": 0.05,
            "disgusted": 0.01,
            "fearful": 0.01,
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

    def test_no_frame_returns_none(
        self, mock_webcam_no_frame, mock_face_detector, mock_classifier
    ):
        recorder = EmotionRecorder(
            webcam=mock_webcam_no_frame,
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

    def test_largest_face_is_selected(
        self, mock_webcam, mock_classifier, fake_frame
    ):
        """When multiple faces are detected, the largest one is used."""
        detector = MagicMock()
        detector.detect.return_value = [
            (10, 10, 50, 50),    # small face (area 2500)
            (100, 100, 200, 200), # large face (area 40000)
        ]
        recorder = EmotionRecorder(
            webcam=mock_webcam,
            face_detector=detector,
            emotion_classifier=mock_classifier,
        )
        recorder.start()
        recorder.record_frame()
        # Verify classifier got the ROI from the larger bounding box
        call_args = mock_classifier.classify.call_args[0][0]
        assert call_args.shape[0] == 200  # height of the large face ROI
        assert call_args.shape[1] == 200  # width of the large face ROI


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
