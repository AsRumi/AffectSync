"""
Transcript alignment module.

Converts Whisper's raw segments (timestamps in seconds from audio start)
into TranscriptSegment objects with millisecond timestamps that align
to the video timeline used by the rest of the pipeline.

No offset correction required, simply multiply each segment timestamps with 1000.

This module also owns JSON export of the transcript.
"""

import json
import logging
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from config import JSON_INDENT, TRANSCRIPT_OUTPUT_DIR
from pipeline.transcriber import RawSegment

logger = logging.getLogger(__name__)


@dataclass
class TranscriptSegment:
    """
    A single transcript segment aligned to the video timeline.

    JSON schema:
        {
            "start_ms": 980,
            "end_ms": 3200,
            "text": "So I was at the DMV the other day..."
        }
    """
    start_ms: int
    end_ms: int
    text: str


class TranscriptAligner:
    """
    Converts Whisper RawSegments to video-timeline TranscriptSegments.

    Usage:
        aligner = TranscriptAligner()
        segments = aligner.align(raw_segments)
        json_path = aligner.export_json(segments, "my_session_transcript.json")
    """

    def align(self, raw_segments: List[RawSegment]) -> List[TranscriptSegment]:
        """
        Convert a list of RawSegments to TranscriptSegments in milliseconds.

        The conversion is: ms = int(seconds * 1000).
        End time is clamped to be at least 1ms after start to guard
        against Whisper segments where start == end.

        Args:
            raw_segments: Segments from Transcriber.transcribe().

        Returns:
            List of TranscriptSegment objects in chronological order.
        """
        aligned: List[TranscriptSegment] = []

        for seg in raw_segments:
            start_ms = int(seg.start_s * 1000)
            end_ms = int(seg.end_s * 1000)

            # End strictly after start
            if end_ms <= start_ms:
                end_ms = start_ms + 1

            aligned.append(TranscriptSegment(
                start_ms=start_ms,
                end_ms=end_ms,
                text=seg.text,
            ))

        logger.info(
            "Aligned %d transcript segments (%.1fs – %.1fs)",
            len(aligned),
            raw_segments[0].start_s if raw_segments else 0.0,
            raw_segments[-1].end_s if raw_segments else 0.0,
        )
        return aligned

    def export_json(
        self,
        segments: List[TranscriptSegment],
        filename: str | None = None,
    ) -> Path:
        """
        Write transcript segments to a JSON file.

        Args:
            segments: List of TranscriptSegment objects to export.
            filename: Output filename. Defaults to 'transcript.json'.

        Returns:
            Path to the written JSON file.

        Raises:
            ValueError: If segments is empty.
        """
        if not segments:
            raise ValueError(
                "No transcript segments to export. "
                "Run transcription first."
            )

        output_dir = Path(TRANSCRIPT_OUTPUT_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)

        filename = filename or "transcript.json"
        output_path = output_dir / filename

        payload = [asdict(seg) for seg in segments]

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=JSON_INDENT, ensure_ascii=False)

        logger.info(
            "Transcript exported — %d segments → %s",
            len(segments),
            output_path,
        )
        return output_path
