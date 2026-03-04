"""
Webcam capture module.

Opens the system webcam and yields BGR frames as numpy arrays.
Handles Windows DirectShow backend and graceful resource cleanup.
"""

import time
from typing import Generator, Tuple

import cv2
import numpy as np

from utils.logger import get_logger
import config

logger = get_logger(__name__)


class WebcamCapture:
    """
    Context-managed webcam interface.

    Usage:
        with WebcamCapture() as cam:
            for timestamp_ms, frame in cam.frames():
                # frame is a BGR numpy array
                process(frame)
    """

    def __init__(
        self,
        device_index: int = config.WEBCAM_INDEX,
        frame_width: int = config.WEBCAM_FRAME_WIDTH,
        frame_height: int = config.WEBCAM_FRAME_HEIGHT,
    ):
        self._device_index = device_index
        self._frame_width = frame_width
        self._frame_height = frame_height
        self._cap: cv2.VideoCapture | None = None
        self._start_time: float = 0.0

    def __enter__(self) -> "WebcamCapture":
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()

    def open(self) -> None:
        """Open the webcam device with the DirectShow backend on Windows."""
        backend = cv2.CAP_DSHOW if config.WEBCAM_BACKEND == "dshow" else cv2.CAP_ANY
        self._cap = cv2.VideoCapture(self._device_index, backend)

        if not self._cap.isOpened():
            raise RuntimeError(
                f"Cannot open webcam at index {self._device_index}. "
                "Check that your camera is connected and not in use by another app."
            )

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._frame_width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._frame_height)
        self._start_time = time.monotonic()

        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(
            "Webcam opened — device=%d, resolution=%dx%d",
            self._device_index, actual_w, actual_h,
        )

    def release(self) -> None:
        """Release the webcam resource."""
        if self._cap is not None and self._cap.isOpened():
            self._cap.release()
            logger.info("Webcam released")
        self._cap = None

    def read_frame(self) -> Tuple[float, np.ndarray]:
        """
        Read a single frame.

        Returns:
            (timestamp_ms, frame) where timestamp_ms is milliseconds since
            the webcam was opened, and frame is a BGR numpy array.

        Raises:
            RuntimeError: If the webcam is not open or the read fails.
        """
        if self._cap is None or not self._cap.isOpened():
            raise RuntimeError("Webcam is not open. Call open() first.")

        ret, frame = self._cap.read()
        if not ret or frame is None:
            raise RuntimeError("Failed to read frame from webcam.")

        timestamp_ms = (time.monotonic() - self._start_time) * 1000.0
        return timestamp_ms, frame

    def frames(self) -> Generator[Tuple[float, np.ndarray], None, None]:
        """
        Yield (timestamp_ms, frame) tuples continuously until the webcam
        is released or an error occurs.
        """
        while self._cap is not None and self._cap.isOpened():
            try:
                yield self.read_frame()
            except RuntimeError as exc:
                logger.warning("Frame read failed: %s", exc)
                break

    @property
    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()
