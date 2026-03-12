"""
Tests for pipeline.video_player.VideoPlayer.

All tests use a synthetic video file generated in a pytest fixture.
No real video files are required.

The synthetic video is a short clip of solid-color frames written via
cv2.VideoWriter at a known FPS, so we can verify metadata extraction
and frame seeking against known values.
"""

import sys
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.video_player import VideoPlayer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_video(tmp_path) -> Path:
    """
    Create a 2-second synthetic video at 10 FPS (20 frames total).

    Frame 0–9:   solid red   (BGR: 0, 0, 255)
    Frame 10–19: solid blue  (BGR: 255, 0, 0)

    Returns the path to the .avi file.
    """
    video_path = tmp_path / "test_clip.avi"
    fps = 10.0
    width, height = 160, 120
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))

    # Red frames (first second)
    red_frame = np.zeros((height, width, 3), dtype=np.uint8)
    red_frame[:, :, 2] = 255  # BGR: red channel
    for _ in range(10):
        writer.write(red_frame)

    # Blue frames (second second)
    blue_frame = np.zeros((height, width, 3), dtype=np.uint8)
    blue_frame[:, :, 0] = 255  # BGR: blue channel
    for _ in range(10):
        writer.write(blue_frame)

    writer.release()
    return video_path


@pytest.fixture
def long_video(tmp_path) -> Path:
    """
    Create a video that exceeds MAX_VIDEO_DURATION_SEC.

    We fake this by patching the config value to a tiny limit (1 second)
    rather than actually generating a 5+ minute video.
    """
    video_path = tmp_path / "long_clip.avi"
    fps = 10.0
    width, height = 160, 120
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))

    frame = np.zeros((height, width, 3), dtype=np.uint8)
    for _ in range(30):  # 3 seconds
        writer.write(frame)

    writer.release()
    return video_path


# ---------------------------------------------------------------------------
# Metadata tests
# ---------------------------------------------------------------------------

class TestVideoPlayerMetadata:

    def test_open_populates_fps(self, synthetic_video):
        player = VideoPlayer(synthetic_video)
        player.open()
        assert player.fps == pytest.approx(10.0, abs=0.5)
        player.release()

    def test_open_populates_frame_count(self, synthetic_video):
        player = VideoPlayer(synthetic_video)
        player.open()
        assert player.frame_count == 20
        player.release()

    def test_open_populates_duration(self, synthetic_video):
        player = VideoPlayer(synthetic_video)
        player.open()
        # 20 frames at 10 FPS = 2000ms
        assert player.duration_ms == pytest.approx(2000.0, abs=200.0)
        player.release()

    def test_open_populates_dimensions(self, synthetic_video):
        player = VideoPlayer(synthetic_video)
        player.open()
        assert player.width == 160
        assert player.height == 120
        player.release()


# ---------------------------------------------------------------------------
# Lifecycle tests
# ---------------------------------------------------------------------------

class TestVideoPlayerLifecycle:

    def test_open_sets_is_open(self, synthetic_video):
        player = VideoPlayer(synthetic_video)
        assert not player.is_open
        player.open()
        assert player.is_open
        player.release()

    def test_release_clears_is_open(self, synthetic_video):
        player = VideoPlayer(synthetic_video)
        player.open()
        player.release()
        assert not player.is_open

    def test_open_nonexistent_file_raises(self, tmp_path):
        player = VideoPlayer(tmp_path / "does_not_exist.mp4")
        with pytest.raises(FileNotFoundError, match="not found"):
            player.open()

    def test_video_path_property(self, synthetic_video):
        player = VideoPlayer(synthetic_video)
        assert player.video_path == synthetic_video


# ---------------------------------------------------------------------------
# Duration validation
# ---------------------------------------------------------------------------

class TestDurationValidation:

    def test_rejects_video_exceeding_max_duration(self, long_video):
        """Patch MAX_VIDEO_DURATION_SEC to 1s so our 3s video is rejected."""
        with patch("pipeline.video_player.MAX_VIDEO_DURATION_SEC", 1):
            player = VideoPlayer(long_video)
            with pytest.raises(ValueError, match="exceeds"):
                player.open()

    def test_accepts_video_within_max_duration(self, synthetic_video):
        """Our 2s synthetic video is well under the default 300s limit."""
        player = VideoPlayer(synthetic_video)
        player.open()  # Should not raise
        assert player.is_open
        player.release()


# ---------------------------------------------------------------------------
# Frame reading tests
# ---------------------------------------------------------------------------

class TestGetFrameAt:

    def test_returns_frame_at_start(self, synthetic_video):
        player = VideoPlayer(synthetic_video)
        player.open()
        frame = player.get_frame_at(0)
        assert frame is not None
        assert frame.shape == (120, 160, 3)
        player.release()

    def test_returns_frame_at_midpoint(self, synthetic_video):
        player = VideoPlayer(synthetic_video)
        player.open()
        frame = player.get_frame_at(1000)  # 1 second in
        assert frame is not None
        assert frame.shape == (120, 160, 3)
        player.release()

    def test_returns_none_past_end(self, synthetic_video):
        player = VideoPlayer(synthetic_video)
        player.open()
        frame = player.get_frame_at(99999)
        assert frame is None
        player.release()

    def test_negative_timestamp_clamps_to_zero(self, synthetic_video):
        player = VideoPlayer(synthetic_video)
        player.open()
        frame = player.get_frame_at(-100)
        assert frame is not None  # Should return first frame
        player.release()

    def test_raises_when_not_open(self, synthetic_video):
        player = VideoPlayer(synthetic_video)
        with pytest.raises(RuntimeError, match="not open"):
            player.get_frame_at(0)


class TestReadNextFrame:

    def test_sequential_read_returns_frames(self, synthetic_video):
        player = VideoPlayer(synthetic_video)
        player.open()
        result = player.read_next_frame()
        assert result is not None
        ts, frame = result
        assert isinstance(ts, float)
        assert frame.shape == (120, 160, 3)
        player.release()

    def test_sequential_read_returns_none_at_end(self, synthetic_video):
        player = VideoPlayer(synthetic_video)
        player.open()
        # Read all 20 frames
        for _ in range(20):
            result = player.read_next_frame()
            assert result is not None
        # Next read should be None — past end
        result = player.read_next_frame()
        assert result is None
        player.release()

    def test_raises_when_not_open(self, synthetic_video):
        player = VideoPlayer(synthetic_video)
        with pytest.raises(RuntimeError, match="not open"):
            player.read_next_frame()


class TestSeekTo:

    def test_seek_then_read(self, synthetic_video):
        player = VideoPlayer(synthetic_video)
        player.open()
        player.seek_to(1000)  # Seek to 1 second
        result = player.read_next_frame()
        assert result is not None
        player.release()

    def test_seek_raises_when_not_open(self, synthetic_video):
        player = VideoPlayer(synthetic_video)
        with pytest.raises(RuntimeError, match="not open"):
            player.seek_to(0)
