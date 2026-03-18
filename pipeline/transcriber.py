"""
Speech-to-text transcription module.

Wraps openai-whisper to transcribe WAV file to time-aligned text segments.
Segments carry start and end time. Converted into milliseconds by TranscriptAligner.

The model is loaded once in __init__ and reused across transcribe() calls. 
Construct at session start and reuse instead of creating per video.

Runs on CPU by default. Uses CUDA if available.
Metrics: 5 mins video -> 2-3 mins post-process.
"""

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

import whisper

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from config import WHISPER_MODEL_SIZE, WHISPER_DEVICE, WHISPER_LANGUAGE

logger = logging.getLogger(__name__)


@dataclass
class RawSegment:
    """
    A single transcription segment as returned by Whisper.

    Timestamps are in seconds from the audio start, use aligner to reconcile with video segments in milliseconds.
    """
    start_s: float
    end_s: float
    text: str


class Transcriber:
    """
    Transcribes a WAV audio file into time-aligned text segments using Whisper.

    Usage:
        transcriber = Transcriber()  # Loads model once
        segments = transcriber.transcribe(wav_path)
        for seg in segments:
            print(seg.start_s, seg.end_s, seg.text)
    """

    def __init__(
        self,
        model_size: str = WHISPER_MODEL_SIZE,
        device: str = WHISPER_DEVICE,
    ):
        self._model_size = model_size
        self._device = device
        self._model = None

    def _load_model(self) -> None:
        """Load the Whisper model into memory if not already loaded."""
        if self._model is not None:
            return

        logger.info(
            "Loading Whisper model '%s' on device '%s' "
            "(first-time download may take a moment)...",
            self._model_size,
            self._device,
        )
        self._model = whisper.load_model(self._model_size, device=self._device)
        logger.info("Whisper model loaded.")

    def transcribe(self, wav_path: str | Path) -> List[RawSegment]:
        """
        Transcribe a WAV file and return time-aligned segments.

        Segments correspond to sentences/natural speech pauses.

        Args:
            wav_path: Path to a 16kHz mono WAV file.

        Returns:
            List of RawSegment objects in chronological order.
            Empty list if no segments detected.

        Raises:
            FileNotFoundError: If the WAV file does not exist.
            RuntimeError: If Whisper fails to process the file.
        """
        wav_path = Path(wav_path)
        if not wav_path.exists():
            raise FileNotFoundError(f"Audio file not found: {wav_path}")

        self._load_model()

        logger.info("Transcribing %s...", wav_path.name)

        try:
            result = self._model.transcribe(
                str(wav_path),
                language=WHISPER_LANGUAGE,   # None = auto-detect
                verbose=False,               # Suppress Whisper's progress output
                fp16=False,                  # fp16 requires CUDA; always use fp32 locally
            )
        except Exception as exc:
            raise RuntimeError(
                f"Whisper transcription failed for {wav_path.name}: {exc}"
            ) from exc

        raw_segments = result.get("segments", [])
        segments = [
            RawSegment(
                start_s=float(seg["start"]),
                end_s=float(seg["end"]),
                text=seg["text"].strip(),
            )
            for seg in raw_segments
            if seg.get("text", "").strip()  # Skipping empty segments
        ]

        logger.info(
            "Transcription complete — %d segments, detected language: %s",
            len(segments),
            result.get("language", "unknown"),
        )
        return segments
