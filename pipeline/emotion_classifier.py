"""
Emotion classification module.
"""

from typing import Dict, Optional, Tuple

import numpy as np
from deepface import DeepFace

from utils.logger import get_logger
import config

logger = get_logger(__name__)


class EmotionClassifier:
    """
    Classify a face crop into emotion labels using DeepFace.

    DeepFace uses a VGG-based FER model by default. On first call it
    downloads the weights (~several MB) and caches them locally.
    """

    def __init__(
        self,
        detector_backend: str = config.FACE_DETECTOR_BACKEND,
        enforce_detection: bool = config.DEEPFACE_ENFORCE_DETECTION,
        confidence_threshold: float = config.EMOTION_CONFIDENCE_THRESHOLD,
    ):
        self._detector_backend = detector_backend
        self._enforce_detection = enforce_detection
        self._confidence_threshold = confidence_threshold
        self._warmed_up = False
        logger.info(
            "EmotionClassifier initialized — backend=%s, threshold=%.2f",
            self._detector_backend, self._confidence_threshold,
        )

    def warm_up(self, dummy_frame: Optional[np.ndarray] = None) -> None:
        """
        Run a throwaway inference to trigger model download and loading.
        Call this once at startup so the first real frame isn't slow.
        """
        if self._warmed_up:
            return

        if dummy_frame is None:
            dummy_frame = np.zeros((48, 48, 3), dtype=np.uint8)

        logger.info("Warming up emotion model (first-time download may take a moment)...")
        try:
            DeepFace.analyze(
                dummy_frame,
                actions=["emotion"],
                detector_backend="skip",
                enforce_detection=False,
                silent=True,
            )
        except Exception as exc:
            logger.warning("Warm-up inference raised (non-fatal): %s", exc)

        self._warmed_up = True
        logger.info("Emotion model warm-up complete")

    def classify(
        self, face_crop: np.ndarray
    ) -> Tuple[str, float, Dict[str, float]]:
        """
        Classify a cropped face image into an emotion.

        Args:
            face_crop: BGR numpy array of the cropped face region.

        Returns:
            (dominant_emotion, confidence, all_scores)

        If confidence is below the threshold, returns ("neutral", ...).
        """
        try:
            results = DeepFace.analyze(
                face_crop,
                actions=["emotion"],
                detector_backend="skip",  # Face is already cropped
                enforce_detection=self._enforce_detection,
                silent=True,
            )
        except Exception as exc:
            logger.warning("Emotion analysis failed: %s", exc)
            return "unknown", 0.0, {}

        # DeepFace returns a list when given a single image
        result = results[0] if isinstance(results, list) else results

        raw_scores: Dict[str, float] = result.get("emotion", {})
        dominant: str = result.get("dominant_emotion", "neutral")

        # Normalize scores to [0, 1] (DeepFace returns percentages 0–100)
        all_scores = {k.lower(): round(v / 100.0, 4) for k, v in raw_scores.items()}
        confidence = all_scores.get(dominant.lower(), 0.0)

        # Fall back to neutral if below threshold
        if confidence < self._confidence_threshold:
            dominant = "neutral"
            confidence = all_scores.get("neutral", 0.0)

        return dominant.lower(), round(confidence, 4), all_scores
