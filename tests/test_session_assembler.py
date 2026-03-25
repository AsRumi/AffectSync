"""
Tests for pipeline.session_assembler.SessionAssembler.

No hardware or model dependencies. All inputs are constructed inline.
Tests cover: schema keys, emotion timeline serialization, transcript
payload serialization, aligned annotation joining, segment position
labeling, gap handling, empty inputs, and JSON file export.
"""

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.session_assembler import SessionAssembler
from pipeline.emotion_recorder import EmotionRecord
from pipeline.transcript_aligner import TranscriptSegment


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def assembler() -> SessionAssembler:
    return SessionAssembler(
        video_source="clip.mp4",
        video_duration_ms=10000,
        viewer_id="test_viewer",
        session_id="fixed-session-id",   # Fixed so tests are deterministic
    )


@pytest.fixture
def emotion_records() -> list[EmotionRecord]:
    return [
        EmotionRecord(
            timestamp_ms=500,
            emotion="neutral",
            confidence=0.70,
            face_detected=True,
            all_scores={"neutral": 0.70, "happy": 0.20, "sad": 0.10},
        ),
        EmotionRecord(
            timestamp_ms=1500,
            emotion="happy",
            confidence=0.87,
            face_detected=True,
            all_scores={"neutral": 0.05, "happy": 0.87, "sad": 0.08},
        ),
        EmotionRecord(
            timestamp_ms=4000,
            emotion="neutral",
            confidence=0.60,
            face_detected=True,
            all_scores={"neutral": 0.60, "happy": 0.30, "sad": 0.10},
        ),
        EmotionRecord(
            timestamp_ms=8000,
            emotion="no_face",
            confidence=0.0,
            face_detected=False,
            all_scores={},
        ),
    ]


@pytest.fixture
def transcript_segments() -> list[TranscriptSegment]:
    return [
        TranscriptSegment(start_ms=0, end_ms=2000, text="Hello world."),
        TranscriptSegment(start_ms=3000, end_ms=6000, text="This is a test."),
        # Gap: 6000–7000 — no segment covers this range
        TranscriptSegment(start_ms=7000, end_ms=10000, text="Final segment."),
    ]


@pytest.fixture
def session(assembler, emotion_records, transcript_segments) -> dict:
    return assembler.assemble(emotion_records, transcript_segments)


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------

class TestSessionSchema:

    def test_top_level_keys_present(self, session):
        expected_keys = {
            "session_id", "video_source", "video_duration_ms",
            "viewer_id", "recorded_at", "emotion_timeline",
            "transcript_segments", "aligned_annotations", "annotated_peaks"
        }
        assert set(session.keys()) == expected_keys

    def test_session_id_is_injected_value(self, session):
        assert session["session_id"] == "fixed-session-id"

    def test_video_source_preserved(self, session):
        assert session["video_source"] == "clip.mp4"

    def test_video_duration_ms_is_int(self, session):
        assert isinstance(session["video_duration_ms"], int)
        assert session["video_duration_ms"] == 10000

    def test_viewer_id_preserved(self, session):
        assert session["viewer_id"] == "test_viewer"

    def test_recorded_at_is_iso_string(self, session):
        # Should be a non-empty ISO-8601 string
        assert isinstance(session["recorded_at"], str)
        assert "T" in session["recorded_at"]  # ISO format contains 'T'

    def test_session_id_is_uuid4_by_default(self):
        """When no session_id is injected, it should be a valid UUID4."""
        import re
        assembler = SessionAssembler(video_source="v.mp4", video_duration_ms=1000)
        session = assembler.assemble([], [])
        uuid_pattern = re.compile(
            r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
        )
        assert uuid_pattern.match(session["session_id"])


# ---------------------------------------------------------------------------
# Emotion timeline tests
# ---------------------------------------------------------------------------

class TestEmotionTimeline:

    def test_emotion_timeline_count_matches_records(self, session, emotion_records):
        assert len(session["emotion_timeline"]) == len(emotion_records)

    def test_emotion_timeline_entry_keys(self, session):
        entry = session["emotion_timeline"][0]
        assert set(entry.keys()) == {
            "timestamp_ms", "emotion", "confidence", "face_detected", "all_scores"
        }

    def test_emotion_timeline_values_correct(self, session):
        entry = session["emotion_timeline"][1]  # happy record
        assert entry["timestamp_ms"] == 1500
        assert entry["emotion"] == "happy"
        assert entry["confidence"] == 0.87
        assert entry["face_detected"] is True
        assert entry["all_scores"]["happy"] == 0.87

    def test_face_detected_false_preserved(self, session):
        # Last record has face_detected=False
        entry = session["emotion_timeline"][3]
        assert entry["face_detected"] is False

    def test_emotion_timeline_empty_when_no_records(self, assembler, transcript_segments):
        session = assembler.assemble([], transcript_segments)
        assert session["emotion_timeline"] == []


# ---------------------------------------------------------------------------
# Transcript segments tests
# ---------------------------------------------------------------------------

class TestTranscriptSegments:

    def test_transcript_segment_count_matches(self, session, transcript_segments):
        assert len(session["transcript_segments"]) == len(transcript_segments)

    def test_transcript_segment_keys(self, session):
        entry = session["transcript_segments"][0]
        assert set(entry.keys()) == {"start_ms", "end_ms", "text"}

    def test_transcript_segment_values_correct(self, session):
        entry = session["transcript_segments"][0]
        assert entry["start_ms"] == 0
        assert entry["end_ms"] == 2000
        assert entry["text"] == "Hello world."

    def test_transcript_segments_empty_when_no_transcript(self, assembler, emotion_records):
        session = assembler.assemble(emotion_records, [])
        assert session["transcript_segments"] == []


# ---------------------------------------------------------------------------
# Aligned annotations tests
# ---------------------------------------------------------------------------

class TestAlignedAnnotations:

    def test_annotation_count_matches_emotion_records(self, session, emotion_records):
        assert len(session["aligned_annotations"]) == len(emotion_records)

    def test_annotation_keys(self, session):
        entry = session["aligned_annotations"][0]
        assert set(entry.keys()) == {
            "timestamp_ms", "emotion", "confidence", "face_detected",
            "transcript_segment", "segment_position",
        }

    def test_timestamp_within_segment_gets_text(self, session):
        # timestamp_ms=500 falls in segment "Hello world." (0–2000)
        entry = session["aligned_annotations"][0]
        assert entry["transcript_segment"] == "Hello world."
        assert entry["segment_position"] is not None

    def test_timestamp_in_gap_gets_none(self, assembler):
        """A timestamp at 6500ms falls in the gap between segments (6000–7000)."""
        records = [EmotionRecord(
            timestamp_ms=6500, emotion="neutral", confidence=0.5,
            face_detected=True, all_scores={},
        )]
        segments = [
            TranscriptSegment(start_ms=3000, end_ms=6000, text="Before gap."),
            TranscriptSegment(start_ms=7000, end_ms=10000, text="After gap."),
        ]
        session = assembler.assemble(records, segments)
        entry = session["aligned_annotations"][0]
        assert entry["transcript_segment"] is None
        assert entry["segment_position"] is None

    def test_timestamp_past_all_segments_gets_none(self, assembler):
        records = [EmotionRecord(
            timestamp_ms=99000, emotion="neutral", confidence=0.5,
            face_detected=True, all_scores={},
        )]
        segments = [TranscriptSegment(start_ms=0, end_ms=2000, text="Only segment.")]
        session = assembler.assemble(records, segments)
        assert session["aligned_annotations"][0]["transcript_segment"] is None

    def test_both_empty_gives_empty_annotations(self, assembler):
        session = assembler.assemble([], [])
        assert session["aligned_annotations"] == []

    def test_empty_transcript_all_annotations_have_none(self, assembler, emotion_records):
        session = assembler.assemble(emotion_records, [])
        for entry in session["aligned_annotations"]:
            assert entry["transcript_segment"] is None
            assert entry["segment_position"] is None


# ---------------------------------------------------------------------------
# Segment position tests
# ---------------------------------------------------------------------------

class TestSegmentPosition:

    def test_position_start(self, assembler):
        """timestamp at 9% of segment duration → "start" (< 20%)."""
        records = [EmotionRecord(
            timestamp_ms=90, emotion="happy", confidence=0.9,
            face_detected=True, all_scores={},
        )]
        segments = [TranscriptSegment(start_ms=0, end_ms=1000, text="Test.")]
        session = assembler.assemble(records, segments)
        assert session["aligned_annotations"][0]["segment_position"] == "start"

    def test_position_mid(self, assembler):
        """timestamp at 50% of segment duration → "mid"."""
        records = [EmotionRecord(
            timestamp_ms=500, emotion="happy", confidence=0.9,
            face_detected=True, all_scores={},
        )]
        segments = [TranscriptSegment(start_ms=0, end_ms=1000, text="Test.")]
        session = assembler.assemble(records, segments)
        assert session["aligned_annotations"][0]["segment_position"] == "mid"

    def test_position_end(self, assembler):
        """timestamp at 90% of segment duration → "end" (>= 80%)."""
        records = [EmotionRecord(
            timestamp_ms=900, emotion="happy", confidence=0.9,
            face_detected=True, all_scores={},
        )]
        segments = [TranscriptSegment(start_ms=0, end_ms=1000, text="Test.")]
        session = assembler.assemble(records, segments)
        assert session["aligned_annotations"][0]["segment_position"] == "end"

    def test_position_at_exact_start_boundary(self, assembler):
        """timestamp == start_ms → fraction = 0.0 → "start"."""
        records = [EmotionRecord(
            timestamp_ms=0, emotion="neutral", confidence=0.5,
            face_detected=True, all_scores={},
        )]
        segments = [TranscriptSegment(start_ms=0, end_ms=1000, text="Test.")]
        session = assembler.assemble(records, segments)
        assert session["aligned_annotations"][0]["segment_position"] == "start"

    def test_position_at_exact_20_percent(self, assembler):
        """timestamp at exactly 20% → "mid" (boundary is exclusive for start)."""
        records = [EmotionRecord(
            timestamp_ms=200, emotion="neutral", confidence=0.5,
            face_detected=True, all_scores={},
        )]
        segments = [TranscriptSegment(start_ms=0, end_ms=1000, text="Test.")]
        session = assembler.assemble(records, segments)
        assert session["aligned_annotations"][0]["segment_position"] == "mid"

    def test_position_at_exact_80_percent(self, assembler):
        """timestamp at exactly 80% → "end" (boundary is inclusive for end)."""
        records = [EmotionRecord(
            timestamp_ms=800, emotion="neutral", confidence=0.5,
            face_detected=True, all_scores={},
        )]
        segments = [TranscriptSegment(start_ms=0, end_ms=1000, text="Test.")]
        session = assembler.assemble(records, segments)
        assert session["aligned_annotations"][0]["segment_position"] == "end"


# ---------------------------------------------------------------------------
# JSON export tests
# ---------------------------------------------------------------------------

class TestExportJson:

    def test_export_creates_file(self, assembler, session, tmp_path):
        with patch("pipeline.session_assembler.SESSION_OUTPUT_DIR", str(tmp_path)):
            path = assembler.export_json(session, "test_session.json")
        assert path.exists()

    def test_export_default_filename(self, assembler, session, tmp_path):
        with patch("pipeline.session_assembler.SESSION_OUTPUT_DIR", str(tmp_path)):
            path = assembler.export_json(session)
        assert path.name == "session.json"

    def test_exported_json_is_valid(self, assembler, session, tmp_path):
        with patch("pipeline.session_assembler.SESSION_OUTPUT_DIR", str(tmp_path)):
            path = assembler.export_json(session, "valid.json")
        with open(path, encoding="utf-8") as f:
            loaded = json.load(f)
        assert isinstance(loaded, dict)

    def test_exported_json_roundtrip(self, assembler, session, tmp_path):
        """The loaded JSON should match the original session dict exactly."""
        with patch("pipeline.session_assembler.SESSION_OUTPUT_DIR", str(tmp_path)):
            path = assembler.export_json(session, "roundtrip.json")
        with open(path, encoding="utf-8") as f:
            loaded = json.load(f)
        assert loaded["session_id"] == session["session_id"]
        assert loaded["video_source"] == session["video_source"]
        assert len(loaded["emotion_timeline"]) == len(session["emotion_timeline"])
        assert len(loaded["transcript_segments"]) == len(session["transcript_segments"])
        assert len(loaded["aligned_annotations"]) == len(session["aligned_annotations"])

    def test_export_empty_session_raises(self, assembler, tmp_path):
        with pytest.raises(ValueError, match="empty"):
            assembler.export_json({})
