"""
Audio extraction module.
ffmpeg-python preferred over moviepy for control over sample rate and channel count.

Extract 16kHz mono WAV file compatible with Whisper.
Output in TEMP_AUDIO_DIR, deletion to be handled by the caller. (Call AudioExtracter.cleanup())
"""

import logging
import sys
from pathlib import Path

import ffmpeg

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from config import AUDIO_SAMPLE_RATE, TEMP_AUDIO_DIR

logger = logging.getLogger(__name__)

class AudioExtractor:
    """
    Extract audio track from video file to temporary WAV.

    Usage:
        extractor = AudioExtractor("clip.mp4")
        wav_path = extractor.extract()
        # ... pass wav_path to Transcriber ...
        extractor.cleanup()
    """

    def __init__(
        self,
        video_path: str | Path,
        sample_rate: int = AUDIO_SAMPLE_RATE,
        output_dir: Path = TEMP_AUDIO_DIR,
    ):
        self._video_path = Path(video_path)
        self._sample_rate = sample_rate
        self._output_dir = Path(output_dir)
        self._wav_path: Path | None = None

    @property
    def wav_path(self) -> Path | None:
        """Path to the extracted WAV, or None if extract() hasn't been called."""
        return self._wav_path

    def extract(self) -> Path:
        """
        Extract audio from the video file and save as a 16kHz mono WAV.

        Output file: parent_path/TEMP_AUDIO_DIR/videoFileName.wav
        
        Returns:
            Path to the extracted WAV file.

        Raises:
            FileNotFoundError: Video file does not exist.
            RuntimeError: ffmpeg filed to extract audio.
        """
        if not self._video_path.exists():
            raise FileNotFoundError(
                f"Video file not found: {self._video_path}"
            )

        self._output_dir.mkdir(parents=True, exist_ok=True)

        wav_filename = self._video_path.stem + ".wav"
        self._wav_path = self._output_dir / wav_filename

        logger.info(
            "Extracting audio from %s → %s (sample_rate=%dHz, mono)",
            self._video_path.name,
            self._wav_path.name,
            self._sample_rate,
        )

        try:
            (
                ffmpeg
                .input(str(self._video_path))
                .output(
                    str(self._wav_path),
                    acodec="pcm_s16le",     # 16-bit PCM, Whisper compatible
                    ac=1,                   # Mono to reduce file size, Whisper handles it
                    ar=self._sample_rate,   # 16kHz
                    vn=None,                # No video stream in output
                )
                .overwrite_output()         # Safe to overwrite temp files
                .run(quiet=True)            # Suppress ffmpeg console output
            )
        except ffmpeg.Error as exc:
            # ffmpeg.Error carries the stderr bytes; decode
            stderr = exc.stderr.decode("utf-8", errors="replace") if exc.stderr else ""
            raise RuntimeError(
                f"ffmpeg failed to extract audio from {self._video_path.name}.\n"
                f"ffmpeg stderr: {stderr}"
            ) from exc

        logger.info(
            "Audio extracted successfully — %s (%.1f KB)",
            self._wav_path.name,
            self._wav_path.stat().st_size / 1024,
        )
        return self._wav_path

    def cleanup(self) -> None:
        """
        Delete the extracted WAV file from disk.

        Safe to call without extract().
        """
        if self._wav_path is not None and self._wav_path.exists():
            self._wav_path.unlink()
            logger.info("Temp audio file deleted — %s", self._wav_path.name)
            self._wav_path = None
