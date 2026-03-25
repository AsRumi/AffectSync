"""
Session dataset assembler.

Merges the emotion timeline (from EmotionRecorder) and transcript segments
(from TranscriptAligner) into the full session JSON defined in the MVP spec.

Find the timestamp for each emotion record (if it exists) and charecterize it in segments.

Then run PeakDetector over emotion_timeline and append annotated_peaks.

Segment thresholds:
    "start" — 0   - 0.2
    "mid"   — 0.2 - 0.8
    "end"   — 0.8 - 1
"""

import json
import logging
import sys
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from config import JSON_INDENT, SESSION_OUTPUT_DIR
from pipeline.emotion_recorder import EmotionRecord
from pipeline.transcript_aligner import TranscriptSegment
from pipeline.peak_detector import PeakDetector

logger = logging.getLogger(__name__)

# Segment position thresholds (fraction of segment duration)
_POSITION_START_THRESHOLD = 0.20
_POSITION_END_THRESHOLD = 0.80


class SessionAssembler:
    """
    Assembles the full AffectSync session JSON from emotion and transcript data.

    Usage:
        assembler = SessionAssembler(video_source="clip.mp4", video_duration_ms=120000)
        session = assembler.assemble(emotion_records, transcript_segments)
        json_path = assembler.export_json(session, "session_001.json")
    """

    def __init__(
        self,
        video_source: str,
        video_duration_ms: float,
        viewer_id: str = "anonymous",
        session_id: Optional[str] = None,
        peak_detector: Optional[PeakDetector] = None,
    ):
        self._video_source = video_source
        self._video_duration_ms = int(video_duration_ms)
        self._viewer_id = viewer_id
        # Allow injection of a fixed session_id for testability; default to UUID4
        self._session_id = session_id or str(uuid.uuid4())
        
        self._peak_detector = peak_detector or PeakDetector()

    def assemble(
        self,
        emotion_records: List[EmotionRecord],
        transcript_segments: List[TranscriptSegment],
    ) -> dict:
        """
        Build the full session dict matching the JSON schema.

        Args:
            emotion_records: Timestamped emotion observations from EmotionRecorder.
            transcript_segments: Aligned transcript segments from TranscriptAligner.

        Returns:
            A dict with keys: session_id, video_source, video_duration_ms,
            viewer_id, recorded_at, emotion_timeline, transcript_segments,
            aligned_annotations, annotated_peaks.
        """
        emotion_timeline = self._build_emotion_timeline(emotion_records)
        transcript_payload = self._build_transcript_payload(transcript_segments)
        aligned_annotations = self._build_aligned_annotations(
            emotion_records, transcript_segments
        )

        session = {
            "session_id": self._session_id,
            "video_source": self._video_source,
            "video_duration_ms": self._video_duration_ms,
            "viewer_id": self._viewer_id,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "emotion_timeline": emotion_timeline,
            "transcript_segments": transcript_payload,
            "aligned_annotations": aligned_annotations,
        }

        session["annotated_peaks"] = self._peak_detector.detect(session)

        logger.info(
            "Session assembled — %d emotion records, %d transcript segments, "
            "%d aligned annotations, %d peaks",
            len(emotion_timeline),
            len(transcript_payload),
            len(aligned_annotations),
            len(session["annotated_peaks"]),
        )
        return session

    def export_json(
        self,
        session: dict,
        filename: Optional[str] = None,
    ) -> Path:
        """
        Write the assembled session dict to a JSON file.

        Args:
            session: The dict returned by assemble().
            filename: Output filename. Defaults to 'session.json'.

        Returns:
            Path to the written JSON file.

        Raises:
            ValueError: If session is empty.
        """
        if not session:
            raise ValueError("Session dict is empty. Call assemble() first.")

        output_dir = Path(SESSION_OUTPUT_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)

        filename = filename or "session.json"
        output_path = output_dir / filename

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(session, f, indent=JSON_INDENT, ensure_ascii=False)

        logger.info(
            "Session JSON exported — %s (%.1f KB)",
            output_path,
            output_path.stat().st_size / 1024,
        )
        return output_path

    # Private helpers
    def _build_emotion_timeline(
        self, records: List[EmotionRecord]
    ) -> List[dict]:
        """Serialize emotion records to the emotion_timeline schema."""
        return [
            {
                "timestamp_ms": r.timestamp_ms,
                "emotion": r.emotion,
                "confidence": r.confidence,
                "face_detected": r.face_detected,
                "all_scores": r.all_scores,
            }
            for r in records
        ]

    def _build_transcript_payload(
        self, segments: List[TranscriptSegment]
    ) -> List[dict]:
        """Serialize transcript segments to the transcript_segments schema."""
        return [asdict(seg) for seg in segments]

    def _build_aligned_annotations(
        self,
        emotion_records: List[EmotionRecord],
        transcript_segments: List[TranscriptSegment],
    ) -> List[dict]:
        """
        Join each emotion record to its overlapping transcript segment.

        For each emotion record, find the transcript segment whose
        [start_ms, end_ms) window contains the record's timestamp.
        If no segment is active at that moment, transcript_segment is
        None and segment_position is None.
        """
        annotations = []

        for record in emotion_records:
            segment = self._find_segment(record.timestamp_ms, transcript_segments)

            if segment is not None:
                position = self._segment_position(record.timestamp_ms, segment)
                transcript_text = segment.text
            else:
                position = None
                transcript_text = None

            annotations.append({
                "timestamp_ms": record.timestamp_ms,
                "emotion": record.emotion,
                "confidence": record.confidence,
                "face_detected": record.face_detected,
                "transcript_segment": transcript_text,
                "segment_position": position,
            })

        return annotations

    def _find_segment(
        self,
        timestamp_ms: int,
        segments: List[TranscriptSegment],
    ) -> Optional[TranscriptSegment]:
        """
        Return the transcript segment whose window contains timestamp_ms.

        Uses half-open interval [start_ms, end_ms). Returns None if the
        timestamp falls in a gap between segments or outside all segments.
        """
        for seg in segments:
            if seg.start_ms <= timestamp_ms < seg.end_ms:
                return seg
        return None

    def _segment_position(
        self,
        timestamp_ms: int,
        segment: TranscriptSegment,
    ) -> str:
        """
        Classify where within a segment a timestamp falls.

        Thresholds (fraction of segment duration):
            "start" — [0%, 20%)
            "mid"   — [20%, 80%)
            "end"   — [80%, 100%]
        """
        duration = segment.end_ms - segment.start_ms
        if duration <= 0:
            return "start"

        fraction = (timestamp_ms - segment.start_ms) / duration

        if fraction < _POSITION_START_THRESHOLD:
            return "start"
        elif fraction < _POSITION_END_THRESHOLD:
            return "mid"
        else:
            return "end"
