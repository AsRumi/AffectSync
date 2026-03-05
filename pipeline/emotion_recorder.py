"""
Emotion recording session orchestrator.

Runs a timed loop: webcam -> face detection -> emotion classification,
collecting timestamped EmotionRecord entries. Exports to CSV on completion.
"""

import csv
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path

from config import (
    EMOTION_LABELS,
    RECORDER_TARGET_FPS,
    RECORDER_OUTPUT_DIR,
)
from utils.timer import SessionTimer
from pipeline.webcam_capture import WebcamCapture
from pipeline.face_detector import FaceDetector
from pipeline.emotion_classifier import EmotionClassifier

logger = logging.getLogger(__name__)

# CSV column order — fixed so downstream tools can rely on it
CSV_COLUMNS = [
    "timestamp_ms",
    "emotion",
    "confidence",
    "face_detected",
] + EMOTION_LABELS


@dataclass
class EmotionRecord:
    """Single timestamped emotion observation."""

    timestamp_ms: int
    emotion: str
    confidence: float
    face_detected: bool
    all_scores: dict[str, float] = field(default_factory=dict)

    def to_csv_row(self) -> dict:
        """Flatten into a dict matching CSV_COLUMNS."""
        row = {
            "timestamp_ms": self.timestamp_ms,
            "emotion": self.emotion,
            "confidence": self.confidence,
            "face_detected": self.face_detected,
        }
        for label in EMOTION_LABELS:
            row[label] = self.all_scores.get(label, 0.0)
        return row


class EmotionRecorder:
    """
    Orchestrates a recording session: captures webcam frames, detects faces,
    classifies emotions, and stores timestamped records.

    Usage:
        recorder = EmotionRecorder()
        recorder.start()
        # ... in a loop, call recorder.record_frame() at target FPS ...
        recorder.stop()
        recorder.export_csv("session_001.csv")
    """

    def __init__(
        self,
        webcam: WebcamCapture | None = None,
        face_detector: FaceDetector | None = None,
        emotion_classifier: EmotionClassifier | None = None,
        timer: SessionTimer | None = None,
    ):
        self._webcam = webcam or WebcamCapture()
        self._face_detector = face_detector or FaceDetector()
        self._classifier = emotion_classifier or EmotionClassifier()
        self._timer = timer or SessionTimer()
        self._records: list[EmotionRecord] = []

    @property
    def records(self) -> list[EmotionRecord]:
        return list(self._records)

    @property
    def timer(self) -> SessionTimer:
        return self._timer

    def start(self) -> None:
        """Begin a recording session. Clears any previous records."""
        self._records = []
        self._timer.start()
        logger.info("Emotion recording session started.")

    def stop(self) -> None:
        """End the recording session."""
        self._timer.stop()
        duration_s = self._timer.elapsed_ms() / 1000
        logger.info(
            "Session stopped. Duration: %.2fs | Frames recorded: %d",
            duration_s,
            len(self._records),
        )

    def record_frame(self) -> EmotionRecord | None:
        """
        Capture one frame, run face detection and emotion classification,
        and append the result to the session records.

        Returns the EmotionRecord if a frame was processed, or None if
        the webcam returned no frame.
        """
        if not self._timer.is_running:
            raise RuntimeError("Cannot record frames before calling start().")

        frame = self._webcam.read_frame()
        if frame is None:
            logger.warning("Webcam returned no frame — skipping.")
            return None

        timestamp_ms = self._timer.elapsed_ms()
        faces = self._face_detector.detect(frame)

        if not faces:
            record = EmotionRecord(
                timestamp_ms=timestamp_ms,
                emotion="no_face",
                confidence=0.0,
                face_detected=False,
            )
            self._records.append(record)
            return record

        # Use the largest detected face (closest to camera)
        x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
        face_roi = frame[y : y + h, x : x + w]

        emotion, confidence, all_scores = self._classifier.classify(face_roi)

        record = EmotionRecord(
            timestamp_ms=timestamp_ms,
            emotion=emotion,
            confidence=confidence,
            face_detected=True,
            all_scores=all_scores,
        )
        self._records.append(record)
        return record

    def export_csv(self, filename: str | None = None) -> Path:
        """
        Write all recorded emotion data to a CSV file.

        Args:
            filename: Output filename. Defaults to 'emotion_session.csv'.

        Returns:
            Path to the written CSV file.
        """
        if not self._records:
            raise ValueError("No records to export. Run a session first.")

        output_dir = Path(RECORDER_OUTPUT_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)

        filename = filename or "emotion_session.csv"
        output_path = output_dir / filename

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            for record in self._records:
                writer.writerow(record.to_csv_row())

        logger.info("Exported %d records to %s", len(self._records), output_path)
        return output_path
