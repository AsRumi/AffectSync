"""
Emotion recording session orchestrator.

Runs a timed loop: webcam -> face detection -> emotion classification,
collecting timestamped EmotionRecord entries. Exports to CSV on completion.

When used inside SyncController, the timer is shared;
the controller owns start/stop/pause/resume. The recorder just reads
elapsed_ms() for timestamps and checks that the timer has been started.
"""

import csv
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

# Ensure project root is importable
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from config import (
    EMOTION_LABELS,
    RECORDER_OUTPUT_DIR,
    EMOTION_CONFIDENCE_THRESHOLD,
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

    Usage (standalone — Phase 1):
        recorder = EmotionRecorder()
        recorder.start()
        # ... in a loop, call recorder.record_frame() at target FPS ...
        recorder.stop()
        recorder.export_csv("session_001.csv")

    Usage (inside SyncController — Phase 2):
        # Controller injects a shared timer and calls record_frame() directly.
        # Controller owns start/stop/pause/resume on the timer.
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
        if not self._webcam.is_open:
            self._webcam.open()
        self._timer.start()
        logger.info("Emotion recording session started.")

    def stop(self) -> None:
        """End the recording session and release the webcam."""
        self._timer.stop()
        self._webcam.release()
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
        the webcam failed to return a frame.

        The timer must be running (or at least started).
        """
        if not (self._timer.is_running or self._timer.is_paused):
            raise RuntimeError("Cannot record frames before calling start().")

        try:
            _webcam_ts, frame = self._webcam.read_frame()
        except RuntimeError:
            logger.warning("Webcam returned no frame — skipping.")
            return None

        # Use our session timer for the timestamp, not the webcam's
        timestamp_ms = self._timer.elapsed_ms()

        # FaceDetector.detect() -> (x,y,w,h) or None
        bbox = self._face_detector.detect(frame)

        if bbox is None:
            record = EmotionRecord(
                timestamp_ms=timestamp_ms,
                emotion="no_face",
                confidence=0.0,
                face_detected=False,
            )
            self._records.append(record)
            return record

        # Crop the face region for the classifier
        face_roi = self._face_detector.crop_face(frame)
        if face_roi is None:
            # detect() found a face but crop_face() didn't — unlikely but safe
            record = EmotionRecord(
                timestamp_ms=timestamp_ms,
                emotion="no_face",
                confidence=0.0,
                face_detected=False,
            )
            self._records.append(record)
            return record

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
