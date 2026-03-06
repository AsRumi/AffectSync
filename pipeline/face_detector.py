"""
Face detection module.

Locates the largest face in a BGR frame and returns the cropped region.
Uses OpenCV's Haar Cascade as the default detector for speed on CPU.
"""

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from utils.logger import get_logger
import config

logger = get_logger(__name__)

# Select haarcascade_frontalface_default.xml from OpenCV
_CASCADE_PATH = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"


class FaceDetector:
    """
    Detect and crop the dominant face from a BGR frame.

    Uses OpenCV Haar Cascade for fast CPU-friendly detection.
    Returns None when no face meets the minimum size threshold.
    """

    def __init__(
        self,
        min_face_size: Tuple[int, int] = config.FACE_MIN_SIZE,
        scale_factor: float = 1.3,
        min_neighbors: int = 5,
    ):
        if not _CASCADE_PATH.exists():
            raise FileNotFoundError(
                f"Haar cascade XML not found at {_CASCADE_PATH}. "
                "Verify your opencv-python installation."
            )

        self._classifier = cv2.CascadeClassifier(str(_CASCADE_PATH))
        self._min_face_size = min_face_size
        self._scale_factor = scale_factor
        self._min_neighbors = min_neighbors
        logger.info("FaceDetector initialized — cascade=%s", _CASCADE_PATH.name)

    def detect(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Detect the largest face bounding box in the frame.

        Args:
            frame: BGR numpy array from the webcam.

        Returns:
            (x, y, w, h) of the largest detected face, or None if no face found.
            
        Hyperparams Info:
        scaleFactor
            The classifier can only detect faces at one size at a time, so it scans the image repeatedly at shrinking scales — each pass makes the image slightly smaller. scaleFactor controls how much smaller each pass is.

            A value of 1.1 means each pass shrinks the image by 10%
            A value of 1.3 means 30% smaller each pass

            A smaller scaleFactor (closer to 1.0) means more passes, more chances to catch a face → higher detection rate but slower. A larger value is faster but might miss faces that fall between scales.

        minNeighbors
            The classifier works by sliding a detection window across the image. It will get many "candidate" hits in roughly the same area. minNeighbors sets how many neighboring candidates must agree before a detection is confirmed as a real face.

            Low value → more detections, but more false positives (random patches flagged as faces)
            High value → stricter, fewer false positives, but might miss real faces

            It's essentially a confidence threshold — a lone candidate hit is probably noise, but 5 overlapping hits in the same spot is likely a real face.

        minSize
            The minimum pixel dimensions (width, height) a detected face must be to count. Any candidate smaller than this is ignored outright.
            This is useful for two reasons:

            Filters out tiny false positives from background noise
            Saves computation time by skipping implausibly small detections

            For a webcam feed where the person is reasonably close, setting something like (60, 60) makes sense — you wouldn't expect a real face to occupy fewer pixels than that.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray) # Improve contrast across entire image (eg. if lighting was poor and you have dark patches dominating leading to lots of values close to 0)

        faces = self._classifier.detectMultiScale(
            gray,
            scaleFactor=self._scale_factor,
            minNeighbors=self._min_neighbors,
            minSize=self._min_face_size,
        )

        if len(faces) == 0:
            return None

        # Return the largest face by area
        largest = max(faces, key=lambda rect: rect[2] * rect[3])
        return tuple(largest)

    def crop_face(
        self, frame: np.ndarray, padding: float = 0.1
    ) -> Optional[np.ndarray]:
        """
        Detect and return the cropped face region with optional padding.

        Args:
            frame: BGR numpy array.
            padding: Fractional padding around the detected box (0.1 = 10%).

        Returns:
            Cropped BGR face image, or None if no face detected.
        """
        bbox = self.detect(frame)
        if bbox is None:
            return None

        x, y, w, h = bbox
        frame_h, frame_w = frame.shape[:2]

        pad_x = int(w * padding)
        pad_y = int(h * padding)

        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(frame_w, x + w + pad_x)
        y2 = min(frame_h, y + h + pad_y)

        return frame[y1:y2, x1:x2] # Splice the numpy array to return only those rows and cols that contain the face, 3 RGB channels carries over automatically, no need to splice
