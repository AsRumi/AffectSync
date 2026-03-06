"""
AffectSync — Central Configuration

All configurable values live here.
"""

from pathlib import Path

# ──────────────────────────────────────────────
# Project Paths
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# ──────────────────────────────────────────────
# Webcam Settings
# ──────────────────────────────────────────────
WEBCAM_INDEX = 0                    # Default camera device index
WEBCAM_BACKEND = "dshow"            # DirectShow backend for Windows
WEBCAM_FRAME_WIDTH = 640
WEBCAM_FRAME_HEIGHT = 480

# ──────────────────────────────────────────────
# Emotion Detection Settings
# ──────────────────────────────────────────────
EMOTION_LABELS = [
    "angry", "disgusted", "fearful", "happy", "sad", "surprised", "neutral"
]
EMOTION_CONFIDENCE_THRESHOLD = 0.30  # Minimum confidence to report a non-neutral label
TARGET_FPS = 10                      # Inference FPS target (not webcam FPS)

# ──────────────────────────────────────────────
# Face Detection Settings
# ──────────────────────────────────────────────
FACE_DETECTOR_BACKEND = "opencv"     # DeepFace detector backend: opencv, mtcnn, retinaface
FACE_MIN_SIZE = (48, 48)             # Minimum face crop size in pixels (w, h)

# ──────────────────────────────────────────────
# DeepFace Settings
# ──────────────────────────────────────────────
DEEPFACE_MODEL_NAME = "Emotion"      # DeepFace action
DEEPFACE_ENFORCE_DETECTION = False   # Set true to raise error when no face found

# ──────────────────────────────────────────────
# Video Playback Settings (Phase 2+)
# ──────────────────────────────────────────────
MAX_VIDEO_DURATION_SEC = 300         # 5 minute cap for MVP clips

# ──────────────────────────────────────────────
# Whisper Transcription Settings (Phase 3+)
# ──────────────────────────────────────────────
WHISPER_MODEL_SIZE = "base"          # Options: tiny, base, small, medium, large

# ──────────────────────────────────────────────
# Dataset Export Settings (Phase 4+)
# ──────────────────────────────────────────────
JSON_INDENT = 2                      # Pretty-print indent for output JSON

# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────
LOG_LEVEL = "INFO"                   # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FORMAT = "%(asctime)s | %(name)-20s | %(levelname)-7s | %(message)s"
LOG_DATE_FORMAT = "%H:%M:%S"

# ──────────────────────────────────────────────
# Emotion Recording Session
# ──────────────────────────────────────────────
RECORDER_TARGET_FPS = 10          # Target frames per second for emotion capture
RECORDER_OUTPUT_DIR = "outputs"   # Directory where session CSVs are saved

# ──────────────────────────────────────────────
# Emotion Labels (canonical order, used across all modules)
# ──────────────────────────────────────────────
EMOTION_LABELS = [
    "happy",
    "sad",
    "angry",
    "surprised",
    "disgusted",
    "fearful",
    "neutral",
]