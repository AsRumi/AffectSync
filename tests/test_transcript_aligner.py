"""
Tests for pipeline.transcript_aligner.TranscriptAligner.

No mocking required — TranscriptAligner is pure Python with no hardware
or model dependencies. The only external interaction is filesystem I/O
for export_json(), which uses tmp_path fixtures.
"""

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.transcript_aligner import TranscriptAligner, TranscriptSegment
from pipeline.transcriber import RawSegment


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def aligner() -> TranscriptAligner:
    return TranscriptAligner()


@pytest.fixture
def raw_segments() -> list[RawSegment]:
    return [
        RawSegment(start_s=0.0, end_s=2.5, text="Hello world."),
        RawSegment(start_s=2.5, end_s=5.0, text="This is a test."),
        RawSegment(start_s=5.0, end_s=8.3, text="One more segment here."),
    ]


# ---------------------------------------------------------------------------
# Alignment tests
# ---------------------------------------------------------------------------

class TestAlign:

    def test_returns_list_of_transcript_segments(self, aligner, raw_segments):
        result = aligner.align(raw_segments)
        assert isinstance(result, list)
        assert all(isinstance(s, TranscriptSegment) for s in result)

    def test_count_matches_input(self, aligner, raw_segments):
        result = aligner.align(raw_segments)
        assert len(result) == len(raw_segments)

    def test_seconds_converted_to_milliseconds(self, aligner):
        segments = [RawSegment(start_s=1.0, end_s=3.5, text="Test.")]
        result = aligner.align(segments)
        assert result[0].start_ms == 1000
        assert result[0].end_ms == 3500

    def test_text_preserved_exactly(self, aligner, raw_segments):
        result = aligner.align(raw_segments)
        for original, aligned in zip(raw_segments, result):
            assert aligned.text == original.text

    def test_fractional_seconds_truncated_not_rounded(self, aligner):
        """int() truncates — 2.999s → 2999ms, not 3000ms."""
        segments = [RawSegment(start_s=0.0, end_s=2.999, text="Edge case.")]
        result = aligner.align(segments)
        assert result[0].end_ms == 2999

    def test_segments_in_chronological_order(self, aligner, raw_segments):
        result = aligner.align(raw_segments)
        start_times = [s.start_ms for s in result]
        assert start_times == sorted(start_times)

    def test_empty_input_returns_empty_list(self, aligner):
        result = aligner.align([])
        assert result == []

    def test_clamps_end_ms_when_equal_to_start_ms(self, aligner):
        """Whisper can rarely produce segments where start == end."""
        segments = [RawSegment(start_s=1.0, end_s=1.0, text="Flash.")]
        result = aligner.align(segments)
        assert result[0].end_ms > result[0].start_ms

    def test_large_timestamp_accuracy(self, aligner):
        """Check ms conversion stays accurate near the 5-minute clip limit."""
        segments = [RawSegment(start_s=299.5, end_s=300.0, text="Final words.")]
        result = aligner.align(segments)
        assert result[0].start_ms == 299500
        assert result[0].end_ms == 300000


# ---------------------------------------------------------------------------
# JSON export tests
# ---------------------------------------------------------------------------

class TestExportJson:

    def test_export_creates_file(self, aligner, raw_segments, tmp_path):
        with patch("pipeline.transcript_aligner.TRANSCRIPT_OUTPUT_DIR", str(tmp_path)):
            segments = aligner.align(raw_segments)
            path = aligner.export_json(segments, "test_transcript.json")
        assert path.exists()

    def test_export_default_filename(self, aligner, raw_segments, tmp_path):
        with patch("pipeline.transcript_aligner.TRANSCRIPT_OUTPUT_DIR", str(tmp_path)):
            segments = aligner.align(raw_segments)
            path = aligner.export_json(segments)
        assert path.name == "transcript.json"

    def test_exported_json_is_valid(self, aligner, raw_segments, tmp_path):
        with patch("pipeline.transcript_aligner.TRANSCRIPT_OUTPUT_DIR", str(tmp_path)):
            segments = aligner.align(raw_segments)
            path = aligner.export_json(segments, "valid.json")

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        assert isinstance(data, list)

    def test_exported_json_has_correct_keys(self, aligner, raw_segments, tmp_path):
        with patch("pipeline.transcript_aligner.TRANSCRIPT_OUTPUT_DIR", str(tmp_path)):
            segments = aligner.align(raw_segments)
            path = aligner.export_json(segments, "keys.json")

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        for entry in data:
            assert "start_ms" in entry
            assert "end_ms" in entry
            assert "text" in entry

    def test_exported_json_row_count_matches(self, aligner, raw_segments, tmp_path):
        with patch("pipeline.transcript_aligner.TRANSCRIPT_OUTPUT_DIR", str(tmp_path)):
            segments = aligner.align(raw_segments)
            path = aligner.export_json(segments, "count.json")

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        assert len(data) == len(raw_segments)

    def test_exported_values_match_segments(self, aligner, tmp_path):
        raw = [RawSegment(start_s=1.0, end_s=3.0, text="Hello.")]
        with patch("pipeline.transcript_aligner.TRANSCRIPT_OUTPUT_DIR", str(tmp_path)):
            segments = aligner.align(raw)
            path = aligner.export_json(segments, "values.json")

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        assert data[0]["start_ms"] == 1000
        assert data[0]["end_ms"] == 3000
        assert data[0]["text"] == "Hello."

    def test_export_empty_segments_raises(self, aligner, tmp_path):
        with pytest.raises(ValueError, match="No transcript segments"):
            aligner.export_json([])
