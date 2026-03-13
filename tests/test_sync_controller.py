"""
Tests for pipeline.sync_controller.SyncController.

All hardware dependencies (webcam, face detector, classifier, video file)
are replaced with mocks via dependency injection. These tests verify the
orchestration logic: timer coordination, emotion/video pairing, pause/resume
behavior, and end-of-video detection.

Mock strategy:
- VideoPlayer: mocked to return synthetic frames from read_next_frame().
  Sequential reads are the primary method; seeking only happens on resume.
- WebcamCapture: same mock pattern as test_emotion_recorder.py.
- FaceDetector / EmotionClassifier: same mocks as Phase 1.
- SessionTimer: use the REAL timer for state-machine tests, since it's
  pure Python with no hardware dependency.
"""

import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.sync_controller import SyncController, SyncedFrame
from pipeline.emotion_recorder import EmotionRecord
from utils.timer import SessionTimer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_frame():
    """A 120x160 BGR frame simulating a video frame."""
    return np.zeros((120, 160, 3), dtype=np.uint8)


@pytest.fixture
def fake_webcam_frame():
    """A 480x640 BGR frame simulating a webcam frame."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def mock_video_player(fake_frame):
    """
    Mock VideoPlayer that returns frames via read_next_frame().

    Simulates a 2-second video at 10 FPS (20 frames). After 20 reads,
    read_next_frame() returns None (end of video). The duration_ms
    property is set to 2000.0 so the timer-based end check also works.
    """
    player = MagicMock()
    player.video_path = Path("test_clip.mp4")
    player.duration_ms = 2000.0
    player.fps = 10.0
    player.frame_count = 20
    player.width = 160
    player.height = 120
    type(player).is_open = PropertyMock(return_value=True)

    # Track how many frames have been read
    read_count = {"n": 0}

    def _read_next():
        read_count["n"] += 1
        if read_count["n"] > 20:
            return None
        # Return (timestamp_ms, frame) — timestamp simulates position
        pos_ms = (read_count["n"] / 10.0) * 1000.0
        return (pos_ms, fake_frame.copy())

    player.read_next_frame.side_effect = _read_next
    player.open.return_value = None
    player.release.return_value = None
    player.seek_to.return_value = None
    return player


@pytest.fixture
def mock_webcam(fake_webcam_frame):
    webcam = MagicMock()
    type(webcam).is_open = PropertyMock(return_value=True)
    webcam.read_frame.return_value = (0.0, fake_webcam_frame)
    return webcam


@pytest.fixture
def mock_face_detector(fake_webcam_frame):
    detector = MagicMock()
    detector.detect.return_value = (100, 100, 150, 150)
    detector.crop_face.return_value = fake_webcam_frame[100:250, 100:250]
    return detector


@pytest.fixture
def mock_classifier():
    classifier = MagicMock()
    classifier.classify.return_value = (
        "happy",
        0.87,
        {
            "angry": 0.01, "disgusted": 0.01, "fearful": 0.01,
            "happy": 0.87, "sad": 0.02, "surprised": 0.05, "neutral": 0.03,
        },
    )
    return classifier


@pytest.fixture
def controller(mock_video_player, mock_webcam, mock_face_detector, mock_classifier):
    """
    Build a SyncController with all mocks injected.

    We patch the VideoPlayer creation inside the constructor by injecting
    the mock after construction.
    """
    timer = SessionTimer()
    ctrl = SyncController(
        video_path="test_clip.mp4",
        webcam=mock_webcam,
        face_detector=mock_face_detector,
        emotion_classifier=mock_classifier,
        timer=timer,
        display_fps=30,
        emotion_fps=10,
    )
    # Replace the video player with our mock
    ctrl._video_player = mock_video_player
    return ctrl


# ---------------------------------------------------------------------------
# Setup / teardown tests
# ---------------------------------------------------------------------------

class TestSyncControllerLifecycle:

    def test_setup_marks_ready(self, controller):
        controller.setup()
        assert controller._is_setup
        controller.teardown()

    def test_start_before_setup_raises(self, controller):
        with pytest.raises(RuntimeError, match="setup"):
            controller.start()

    def test_run_before_start_raises(self, controller):
        controller.setup()
        with pytest.raises(RuntimeError, match="start"):
            next(controller.run())
        controller.teardown()

    def test_teardown_releases_resources(self, controller, mock_video_player, mock_webcam):
        controller.setup()
        controller.start()
        controller.teardown()
        mock_video_player.release.assert_called_once()
        mock_webcam.release.assert_called()


# ---------------------------------------------------------------------------
# Sync loop tests
# ---------------------------------------------------------------------------

class TestSyncLoop:

    def test_yields_synced_frames(self, controller):
        controller.setup()
        controller.start()

        frames_collected = []
        for synced in controller.run():
            frames_collected.append(synced)
            if len(frames_collected) >= 5:
                controller.stop()
                break

        assert len(frames_collected) == 5
        assert all(isinstance(f, SyncedFrame) for f in frames_collected)
        controller.teardown()

    def test_video_frame_is_present(self, controller):
        controller.setup()
        controller.start()

        synced = next(controller.run())
        assert synced.video_frame is not None
        assert synced.video_frame.shape == (120, 160, 3)

        controller.stop()
        controller.teardown()

    def test_first_frame_runs_emotion_inference(self, controller):
        """Tick 0 should always be an inference frame."""
        controller.setup()
        controller.start()

        synced = next(controller.run())
        assert synced.is_inference_frame is True
        assert synced.emotion_record is not None
        assert synced.emotion_record.emotion == "happy"

        controller.stop()
        controller.teardown()

    def test_non_inference_frames_carry_last_emotion(self, controller):
        """
        Between inference frames, the last emotion record should persist.
        With display_fps=30 and emotion_fps=10, every 3rd frame is inference.
        """
        controller.setup()
        controller.start()

        frames = []
        for synced in controller.run():
            frames.append(synced)
            if len(frames) >= 6:
                controller.stop()
                break

        # Frame 0: inference (tick 0 % 3 == 0)
        assert frames[0].is_inference_frame is True
        # Frame 1: not inference, but should still carry the emotion
        assert frames[1].is_inference_frame is False
        assert frames[1].emotion_record is not None
        assert frames[1].emotion_record.emotion == "happy"

        controller.teardown()

    def test_video_timestamps_increase(self, controller):
        controller.setup()
        controller.start()

        timestamps = []
        for synced in controller.run():
            timestamps.append(synced.video_timestamp_ms)
            if len(timestamps) >= 10:
                controller.stop()
                break

        assert timestamps == sorted(timestamps)
        controller.teardown()

    def test_video_end_stops_loop(self, controller, mock_video_player):
        """When read_next_frame returns None, the loop should exit."""
        # Make the video "end" after 3 frames
        call_count = {"n": 0}
        fake = np.zeros((120, 160, 3), dtype=np.uint8)

        def _read_next():
            call_count["n"] += 1
            if call_count["n"] > 3:
                return None
            return (call_count["n"] * 100.0, fake.copy())

        mock_video_player.read_next_frame.side_effect = _read_next
        # Set duration high so the timer check doesn't trigger first
        mock_video_player.duration_ms = 999999.0

        controller.setup()
        controller.start()

        frames = list(controller.run())
        assert len(frames) == 3
        controller.teardown()

    def test_uses_read_next_frame_not_seek(self, controller, mock_video_player):
        """Verify the loop uses sequential reads, not get_frame_at seeking."""
        controller.setup()
        controller.start()

        count = 0
        for _ in controller.run():
            count += 1
            if count >= 5:
                controller.stop()
                break

        # read_next_frame should have been called, not get_frame_at
        assert mock_video_player.read_next_frame.call_count == 5
        mock_video_player.get_frame_at.assert_not_called()
        controller.teardown()


# ---------------------------------------------------------------------------
# Pause / resume tests
# ---------------------------------------------------------------------------

class TestSyncPauseResume:

    def test_pause_freezes_timer(self, controller):
        controller.setup()
        controller.start()
        time.sleep(0.05)
        controller.pause()
        assert controller.timer.is_paused

        frozen = controller.timer.elapsed_ms()
        time.sleep(0.05)
        assert controller.timer.elapsed_ms() == frozen

        controller.teardown()

    def test_resume_continues_timer(self, controller):
        controller.setup()
        controller.start()
        time.sleep(0.05)
        controller.pause()
        frozen = controller.timer.elapsed_ms()

        controller.resume()
        time.sleep(0.05)
        assert controller.timer.elapsed_ms() > frozen
        assert controller.timer.is_running

        controller.teardown()

    def test_resume_seeks_video_player(self, controller, mock_video_player):
        controller.setup()
        controller.start()
        time.sleep(0.05)
        controller.pause()
        controller.resume()

        mock_video_player.seek_to.assert_called()
        controller.teardown()

    def test_synced_records_property(self, controller):
        controller.setup()
        controller.start()

        count = 0
        for _ in controller.run():
            count += 1
            if count >= 3:
                controller.stop()
                break

        assert len(controller.synced_records) == 3
        controller.teardown()


# ---------------------------------------------------------------------------
# CSV export delegation tests
# ---------------------------------------------------------------------------

class TestSyncExport:

    def test_export_delegates_to_recorder(self, controller, tmp_path):
        with patch("pipeline.emotion_recorder.RECORDER_OUTPUT_DIR", str(tmp_path)):
            controller.setup()
            controller.start()

            # Collect some frames to populate records
            count = 0
            for _ in controller.run():
                count += 1
                if count >= 5:
                    controller.stop()
                    break

            csv_path = controller.export_session_csv("sync_test.csv")
            assert csv_path.exists()
            assert csv_path.name == "sync_test.csv"

            controller.teardown()
