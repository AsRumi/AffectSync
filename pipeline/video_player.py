"""
Video frame server for synchronized playback.

Opens a video file via cv2.VideoCapture and serves frames on demand.
SyncController asks for a frame at a given timestamp,
this module seeks to that position and returns the BGR frame.

Key design decisions:
- cv2.VideoCapture on a file is a sequential frame reader. Seeking via
  CAP_PROP_POS_MSEC is supported but can be inaccurate on some codecs.
  For the MVP (clips < 5 min, forward-only playback), we read frames
  sequentially and only seek when resuming from a pause.
- Duration validation happens at open() time so we fail fast on clips
  that exceed MAX_VIDEO_DURATION_SEC.
- The open/release lifecycle mirrors WebcamCapture for consistency.
"""

import logging
import sys
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from config import MAX_VIDEO_DURATION_SEC

logger = logging.getLogger(__name__)


class VideoPlayer:
    """
    Serves video frames by timestamp from a local video file.

    Usage:
        player = VideoPlayer("clip.mp4")
        player.open()
        frame = player.get_frame_at(1500)  # frame at 1.5 seconds
        player.release()
    """

    def __init__(self, video_path: str | Path):
        self._video_path = Path(video_path)
        self._cap: cv2.VideoCapture | None = None

        # Metadata populated on open()
        self._fps: float = 0.0
        self._frame_count: int = 0
        self._duration_ms: float = 0.0
        self._width: int = 0
        self._height: int = 0

    def open(self) -> None:
        """
        Open the video file and read metadata.

        Raises FileNotFoundError if the file does not exist.
        Raises RuntimeError if OpenCV cannot open the file.
        Raises ValueError if the video exceeds MAX_VIDEO_DURATION_SEC.
        """
        if not self._video_path.exists():
            raise FileNotFoundError(f"Video file not found: {self._video_path}")

        self._cap = cv2.VideoCapture(str(self._video_path))

        if not self._cap.isOpened():
            raise RuntimeError(
                f"Cannot open video file: {self._video_path}. "
                "Check that the codec is supported by your OpenCV build."
            )

        self._fps = self._cap.get(cv2.CAP_PROP_FPS)
        self._frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if self._fps > 0 and self._frame_count > 0:
            self._duration_ms = (self._frame_count / self._fps) * 1000.0
        else:
            self._duration_ms = 0.0

        # Validate duration
        duration_sec = self._duration_ms / 1000.0
        if duration_sec > MAX_VIDEO_DURATION_SEC:
            self._cap.release()
            self._cap = None
            raise ValueError(
                f"Video duration ({duration_sec:.1f}s) exceeds the "
                f"{MAX_VIDEO_DURATION_SEC}s MVP limit. Use a shorter clip."
            )

        logger.info(
            "Video opened — %s | %.1f FPS | %d frames | %.1fs | %dx%d",
            self._video_path.name,
            self._fps,
            self._frame_count,
            duration_sec,
            self._width,
            self._height,
        )

    def release(self) -> None:
        """Release the video file handle."""
        if self._cap is not None and self._cap.isOpened():
            self._cap.release()
            logger.info("Video released — %s", self._video_path.name)
        self._cap = None

    def get_frame_at(self, timestamp_ms: int) -> np.ndarray | None:
        """
        Return the video frame closest to the given timestamp.

        Seeks to the target position using CAP_PROP_POS_MSEC then reads
        one frame. Returns None if the timestamp is past the end of the
        video or if the read fails.

        Args:
            timestamp_ms: Target position in milliseconds from video start.

        Returns:
            BGR numpy array of the frame, or None if unavailable.
        """
        if self._cap is None or not self._cap.isOpened():
            raise RuntimeError("Video is not open. Call open() first.")

        if timestamp_ms < 0:
            timestamp_ms = 0

        if self._duration_ms > 0 and timestamp_ms > self._duration_ms:
            return None

        self._cap.set(cv2.CAP_PROP_POS_MSEC, float(timestamp_ms))
        ret, frame = self._cap.read()

        if not ret or frame is None:
            return None

        return frame

    def read_next_frame(self) -> Tuple[float, np.ndarray] | None:
        """
        Read the next sequential frame without seeking.

        This is faster than get_frame_at() for forward-only playback
        because it avoids the seek overhead. Returns (timestamp_ms, frame)
        or None if we've reached the end of the video.
        """
        if self._cap is None or not self._cap.isOpened():
            raise RuntimeError("Video is not open. Call open() first.")

        ret, frame = self._cap.read()
        if not ret or frame is None:
            return None

        # Read the position AFTER the read; this is where we just read from
        position_ms = self._cap.get(cv2.CAP_PROP_POS_MSEC)
        return position_ms, frame

    def seek_to(self, timestamp_ms: int) -> None:
        """
        Seek to a specific position without reading a frame.

        Useful after resuming from a pause; seek to where the timer
        says we are, then continue with read_next_frame().
        """
        if self._cap is None or not self._cap.isOpened():
            raise RuntimeError("Video is not open. Call open() first.")

        self._cap.set(cv2.CAP_PROP_POS_MSEC, float(max(0, timestamp_ms)))

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def duration_ms(self) -> float:
        return self._duration_ms

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    @property
    def video_path(self) -> Path:
        return self._video_path
