"""
Phase 0 Integration Test Script

Opens the webcam, detects faces, classifies emotions, and logs results
to the console in real time. Press 'q' in the preview window to quit.

Usage:
    python scripts/test_webcam_emotion.py
    python scripts/test_webcam_emotion.py --no-preview   # headless, no window
"""

import argparse
import sys
import time
from pathlib import Path

import cv2

# Ensure project root is on the path
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

import config
from utils.logger import get_logger
from pipeline.webcam_capture import WebcamCapture
from pipeline.face_detector import FaceDetector
from pipeline.emotion_classifier import EmotionClassifier

logger = get_logger("test_webcam_emotion")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AffectSync — Phase 0 webcam emotion test")
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Run without opening a preview window (log-only mode).",
    )
    return parser.parse_args()


def draw_overlay(
    frame, bbox, emotion: str, confidence: float
) -> None:
    """Draw face bounding box and emotion label on the frame."""
    if bbox is None:
        cv2.putText(
            frame, "No face detected", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
        )
        return

    x, y, w, h = bbox
    color = (0, 255, 0)
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    label = f"{emotion} ({confidence:.0%})"
    cv2.putText(
        frame, label, (x, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2,
    )


def run(show_preview: bool = True) -> None:
    logger.info("=== AffectSync Phase 0 — Webcam Emotion Test ===")

    detector = FaceDetector()
    classifier = EmotionClassifier()
    classifier.warm_up()

    frame_interval = 1.0 / config.TARGET_FPS
    frame_count = 0
    inference_count = 0

    # Cache the last bbox and emotion detected to display between inference frame pauses too
    last_bbox = None
    last_emotion = "no_face"
    last_confidence = 0.0

    with WebcamCapture() as cam:
        logger.info(
            "Streaming at target %d FPS. Press 'q' to quit.",
            config.TARGET_FPS,
        )
        last_inference_time = 0.0

        for timestamp_ms, frame in cam.frames():
            frame_count += 1
            now = time.monotonic()

            # Run inference only at TARGET_FPS, but display every frame
            if now - last_inference_time >= frame_interval:
                last_inference_time = now

                # Detect face
                last_bbox = detector.detect(frame)
                face_crop = detector.crop_face(frame) if last_bbox is not None else None

                # Classify emotion
                if face_crop is not None:
                    last_emotion, last_confidence, all_scores = classifier.classify(face_crop)
                    inference_count += 1
                    logger.info(
                        "t=%7.0fms | %-10s | conf=%.2f | scores=%s",
                        timestamp_ms, last_emotion, last_confidence, all_scores,
                    )
                else:
                    last_emotion, last_confidence = "no_face", 0.0
                    logger.debug("t=%7.0fms | no face detected", timestamp_ms)

            # Draw the cached overlay on EVERY frame, then display
            if show_preview:
                draw_overlay(frame, last_bbox, last_emotion, last_confidence)
                cv2.imshow("AffectSync — Phase 0", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    if show_preview:
        cv2.destroyAllWindows()

    logger.info(
        "Session complete — %d frames captured, %d inferences run",
        frame_count, inference_count,
    )


if __name__ == "__main__":
    args = parse_args()
    run(show_preview=not args.no_preview)