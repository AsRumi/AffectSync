"""
Tests for pipeline.peak_detector.PeakDetector.

No hardware or model dependencies. All inputs are constructed inline as
plain dicts matching the emotion_timeline schema used in the session JSON.

Test coverage:
  - Schema: output keys, peak_type values, required fields
  - Onset detection: baseline window, threshold filtering, no baseline present
  - Sustained detection: minimum duration, one-frame interruption tolerance,
    run that is too short, emotion mismatch breaks run
  - Spike detection: above/below threshold, merge gap logic
  - Empty / all-neutral inputs
  - Transcript index lookup
  - Integration: detect() called on a full minimal session dict
  - Session assembler: annotated_peaks key present after assemble()
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.peak_detector import PeakDetector
from pipeline.session_assembler import SessionAssembler
from pipeline.emotion_recorder import EmotionRecord
from pipeline.transcript_aligner import TranscriptSegment


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _frame(timestamp_ms: int, emotion: str, confidence: float) -> dict:
    """Build a minimal emotion_timeline entry."""
    return {
        "timestamp_ms": timestamp_ms,
        "emotion": emotion,
        "confidence": confidence,
        "face_detected": emotion not in ("no_face", "unknown"),
        "all_scores": {},
    }


def _annotation(timestamp_ms: int, transcript_segment: str | None) -> dict:
    """Build a minimal aligned_annotations entry."""
    return {
        "timestamp_ms": timestamp_ms,
        "emotion": "neutral",
        "confidence": 0.5,
        "face_detected": True,
        "transcript_segment": transcript_segment,
        "segment_position": "mid",
    }


def _session(timeline: list, annotations: list | None = None) -> dict:
    """Wrap a timeline and optional annotations into a minimal session dict."""
    return {
        "emotion_timeline": timeline,
        "aligned_annotations": annotations or [],
    }


def _detector(**kwargs) -> PeakDetector:
    """
    Build a PeakDetector with tight thresholds so test timelines don't need
    to be very long. Defaults can be overridden via kwargs.
    """
    defaults = dict(
        confidence_threshold=0.60,
        onset_window_ms=500,
        sustained_min_ms=300,
        spike_threshold=0.80,
        merge_gap_ms=150,
    )
    defaults.update(kwargs)
    return PeakDetector(**defaults)


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------

class TestPeakSchema:

    def test_peak_has_required_keys(self):
        timeline = [
            _frame(0, "neutral", 0.70),
            _frame(100, "neutral", 0.65),
            _frame(600, "happy", 0.85),  # onset after 500ms baseline window
        ]
        peaks = _detector().detect(_session(timeline))
        assert len(peaks) >= 1
        peak = peaks[0]
        expected_keys = {
            "peak_type", "emotion", "start_ms", "end_ms",
            "duration_ms", "peak_confidence", "transcript_segment",
        }
        assert set(peak.keys()) == expected_keys

    def test_peak_type_values_are_valid(self):
        timeline = [
            _frame(0, "neutral", 0.70),
            _frame(100, "neutral", 0.65),
            _frame(600, "happy", 0.90),
        ]
        peaks = _detector().detect(_session(timeline))
        valid_types = {"onset", "sustained", "spike"}
        for peak in peaks:
            assert peak["peak_type"] in valid_types

    def test_duration_ms_is_zero_for_onset(self):
        timeline = [
            _frame(0, "neutral", 0.70),
            _frame(100, "neutral", 0.65),
            _frame(600, "happy", 0.85),
        ]
        peaks = _detector(spike_threshold=0.99).detect(_session(timeline))
        onset_peaks = [p for p in peaks if p["peak_type"] == "onset"]
        assert len(onset_peaks) == 1
        assert onset_peaks[0]["duration_ms"] == 0
        assert onset_peaks[0]["start_ms"] == onset_peaks[0]["end_ms"]

    def test_duration_ms_nonzero_for_sustained(self):
        timeline = [
            _frame(0, "happy", 0.70),
            _frame(100, "happy", 0.72),
            _frame(200, "happy", 0.75),
            _frame(300, "happy", 0.73),
            _frame(400, "happy", 0.71),
        ]
        peaks = _detector(sustained_min_ms=300).detect(_session(timeline))
        sustained = [p for p in peaks if p["peak_type"] == "sustained"]
        assert len(sustained) == 1
        assert sustained[0]["duration_ms"] == 400  # 400 - 0
        assert sustained[0]["end_ms"] > sustained[0]["start_ms"]

    def test_peak_confidence_is_rounded(self):
        timeline = [
            _frame(0, "neutral", 0.70),
            _frame(600, "happy", 0.876543),
        ]
        peaks = _detector(spike_threshold=0.99).detect(_session(timeline))
        for peak in peaks:
            # Should have at most 4 decimal places
            assert peak["peak_confidence"] == round(peak["peak_confidence"], 4)

    def test_peaks_sorted_by_start_ms(self):
        # Two separate onset windows
        timeline = [
            _frame(0, "neutral", 0.70),
            _frame(100, "neutral", 0.65),
            _frame(600, "happy", 0.85),
            _frame(700, "neutral", 0.70),
            _frame(800, "neutral", 0.65),
            _frame(1400, "sad", 0.75),
        ]
        peaks = _detector(spike_threshold=0.99).detect(_session(timeline))
        start_times = [p["start_ms"] for p in peaks]
        assert start_times == sorted(start_times)


# ---------------------------------------------------------------------------
# Onset detection tests
# ---------------------------------------------------------------------------

class TestOnsetDetection:

    def test_detects_onset_after_neutral_baseline(self):
        timeline = [
            _frame(0, "neutral", 0.70),
            _frame(100, "neutral", 0.65),
            _frame(600, "happy", 0.85),
        ]
        peaks = _detector(spike_threshold=0.99).detect(_session(timeline))
        onsets = [p for p in peaks if p["peak_type"] == "onset"]
        assert len(onsets) == 1
        assert onsets[0]["emotion"] == "happy"
        assert onsets[0]["start_ms"] == 600

    def test_detects_onset_after_no_face_baseline(self):
        """no_face counts as a neutral baseline for onset purposes."""
        timeline = [
            _frame(0, "no_face", 0.0),
            _frame(100, "no_face", 0.0),
            _frame(600, "surprised", 0.82),
        ]
        peaks = _detector(spike_threshold=0.99).detect(_session(timeline))
        onsets = [p for p in peaks if p["peak_type"] == "onset"]
        assert len(onsets) == 1
        assert onsets[0]["emotion"] == "surprised"

    def test_no_onset_when_below_confidence_threshold(self):
        timeline = [
            _frame(0, "neutral", 0.70),
            _frame(600, "happy", 0.40),  # Below threshold of 0.60
        ]
        peaks = _detector(spike_threshold=0.99).detect(_session(timeline))
        onsets = [p for p in peaks if p["peak_type"] == "onset"]
        assert len(onsets) == 0

    def test_no_onset_without_baseline_window(self):
        """If there are no frames in the onset window, no onset is generated."""
        timeline = [
            _frame(600, "happy", 0.85),  # No preceding frames within window
        ]
        peaks = _detector(onset_window_ms=500, spike_threshold=0.99).detect(
            _session(timeline)
        )
        onsets = [p for p in peaks if p["peak_type"] == "onset"]
        assert len(onsets) == 0

    def test_no_onset_when_preceded_by_non_baseline(self):
        """If the preceding window contains a non-neutral emotion, no onset."""
        timeline = [
            _frame(0, "happy", 0.70),   # Non-baseline in window
            _frame(400, "happy", 0.85),
        ]
        peaks = _detector(onset_window_ms=500, spike_threshold=0.99).detect(
            _session(timeline)
        )
        onsets = [p for p in peaks if p["peak_type"] == "onset"]
        assert len(onsets) == 0

    def test_onset_emotion_matches_frame(self):
        timeline = [
            _frame(0, "neutral", 0.70),
            _frame(600, "sad", 0.75),
        ]
        peaks = _detector(spike_threshold=0.99).detect(_session(timeline))
        onsets = [p for p in peaks if p["peak_type"] == "onset"]
        assert onsets[0]["emotion"] == "sad"

    def test_multiple_onsets_detected(self):
        """Two separate onset windows produce two onsets."""
        timeline = [
            _frame(0, "neutral", 0.70),
            _frame(100, "neutral", 0.65),
            _frame(600, "happy", 0.85),   # onset 1
            _frame(700, "neutral", 0.70),
            _frame(800, "neutral", 0.65),
            _frame(1400, "sad", 0.75),    # onset 2
        ]
        peaks = _detector(spike_threshold=0.99).detect(_session(timeline))
        onsets = [p for p in peaks if p["peak_type"] == "onset"]
        assert len(onsets) == 2


# ---------------------------------------------------------------------------
# Sustained peak detection tests
# ---------------------------------------------------------------------------

class TestSustainedDetection:

    def test_detects_sustained_run(self):
        timeline = [
            _frame(0, "happy", 0.70),
            _frame(100, "happy", 0.72),
            _frame(200, "happy", 0.75),
            _frame(300, "happy", 0.73),
            _frame(400, "happy", 0.71),
        ]
        peaks = _detector(sustained_min_ms=300).detect(_session(timeline))
        sustained = [p for p in peaks if p["peak_type"] == "sustained"]
        assert len(sustained) == 1
        assert sustained[0]["emotion"] == "happy"
        assert sustained[0]["start_ms"] == 0
        assert sustained[0]["end_ms"] == 400

    def test_run_too_short_not_detected(self):
        timeline = [
            _frame(0, "happy", 0.70),
            _frame(100, "happy", 0.72),
            _frame(200, "neutral", 0.80),  # Run ends here, duration=100ms < 300ms
        ]
        peaks = _detector(sustained_min_ms=300).detect(_session(timeline))
        sustained = [p for p in peaks if p["peak_type"] == "sustained"]
        assert len(sustained) == 0

    def test_emotion_mismatch_breaks_run(self):
        """Switching from happy to sad should end the happy run."""
        timeline = [
            _frame(0, "happy", 0.70),
            _frame(100, "happy", 0.72),
            _frame(200, "sad", 0.70),  # Different emotion — breaks run
            _frame(300, "sad", 0.72),
            _frame(400, "sad", 0.73),
            _frame(500, "sad", 0.71),
        ]
        peaks = _detector(sustained_min_ms=200).detect(_session(timeline))
        sustained = [p for p in peaks if p["peak_type"] == "sustained"]
        emotions = {p["emotion"] for p in sustained}
        # happy run is only 100ms (too short); sad run is 300ms (long enough)
        assert "happy" not in emotions
        assert "sad" in emotions

    def test_one_frame_interruption_tolerated(self):
        """A single no_face frame in the middle of a run should not break it."""
        timeline = [
            _frame(0, "happy", 0.70),
            _frame(100, "happy", 0.72),
            _frame(200, "no_face", 0.0),   # Interruption — tolerated
            _frame(300, "happy", 0.73),
            _frame(400, "happy", 0.71),
        ]
        peaks = _detector(sustained_min_ms=300).detect(_session(timeline))
        sustained = [p for p in peaks if p["peak_type"] == "sustained"]
        assert len(sustained) == 1
        assert sustained[0]["start_ms"] == 0
        assert sustained[0]["end_ms"] == 400

    def test_two_consecutive_interruptions_break_run(self):
        """Two consecutive baseline frames end the run."""
        timeline = [
            _frame(0, "happy", 0.70),
            _frame(100, "happy", 0.72),
            _frame(200, "no_face", 0.0),   # First interruption
            _frame(300, "neutral", 0.80),  # Second — breaks run
            _frame(400, "happy", 0.73),
            _frame(500, "happy", 0.71),
        ]
        peaks = _detector(sustained_min_ms=300).detect(_session(timeline))
        sustained = [p for p in peaks if p["peak_type"] == "sustained"]
        # First run: 0–100ms = 100ms, too short
        # Second run: 400–500ms = 100ms, too short
        assert len(sustained) == 0

    def test_sustained_below_confidence_threshold_not_detected(self):
        timeline = [
            _frame(0, "happy", 0.40),   # Below 0.60 threshold
            _frame(100, "happy", 0.45),
            _frame(200, "happy", 0.42),
            _frame(300, "happy", 0.44),
        ]
        peaks = _detector(sustained_min_ms=200).detect(_session(timeline))
        sustained = [p for p in peaks if p["peak_type"] == "sustained"]
        assert len(sustained) == 0

    def test_sustained_peak_confidence_is_highest_in_run(self):
        timeline = [
            _frame(0, "happy", 0.65),
            _frame(100, "happy", 0.91),  # Peak
            _frame(200, "happy", 0.70),
            _frame(300, "happy", 0.68),
        ]
        peaks = _detector(sustained_min_ms=200).detect(_session(timeline))
        sustained = [p for p in peaks if p["peak_type"] == "sustained"]
        assert len(sustained) == 1
        assert sustained[0]["peak_confidence"] == pytest.approx(0.91, abs=0.001)


# ---------------------------------------------------------------------------
# Spike detection tests
# ---------------------------------------------------------------------------

class TestSpikeDetection:

    def test_detects_spike_above_threshold(self):
        timeline = [
            _frame(0, "happy", 0.92),
        ]
        peaks = _detector(spike_threshold=0.80).detect(_session(timeline))
        spikes = [p for p in peaks if p["peak_type"] == "spike"]
        assert len(spikes) == 1
        assert spikes[0]["emotion"] == "happy"
        assert spikes[0]["start_ms"] == 0

    def test_no_spike_below_threshold(self):
        timeline = [
            _frame(0, "happy", 0.75),  # Below 0.80 spike threshold
        ]
        peaks = _detector(spike_threshold=0.80).detect(_session(timeline))
        spikes = [p for p in peaks if p["peak_type"] == "spike"]
        assert len(spikes) == 0

    def test_baseline_emotion_never_produces_spike(self):
        timeline = [
            _frame(0, "neutral", 0.95),
            _frame(100, "no_face", 0.0),
        ]
        peaks = _detector(spike_threshold=0.80).detect(_session(timeline))
        spikes = [p for p in peaks if p["peak_type"] == "spike"]
        assert len(spikes) == 0

    def test_nearby_spikes_same_emotion_merged(self):
        """Two happy spikes within merge_gap_ms should become one."""
        timeline = [
            _frame(0, "happy", 0.85),
            _frame(100, "happy", 0.92),  # 100ms gap, within merge_gap_ms=150
        ]
        peaks = _detector(spike_threshold=0.80, merge_gap_ms=150).detect(
            _session(timeline)
        )
        spikes = [p for p in peaks if p["peak_type"] == "spike"]
        assert len(spikes) == 1
        # Should keep the higher-confidence one
        assert spikes[0]["peak_confidence"] == pytest.approx(0.92, abs=0.001)

    def test_spikes_different_emotion_not_merged(self):
        """Two spikes of different emotions should not be merged."""
        timeline = [
            _frame(0, "happy", 0.85),
            _frame(50, "sad", 0.90),
        ]
        peaks = _detector(spike_threshold=0.80, merge_gap_ms=150).detect(
            _session(timeline)
        )
        spikes = [p for p in peaks if p["peak_type"] == "spike"]
        assert len(spikes) == 2

    def test_spikes_outside_merge_gap_kept_separate(self):
        """Two happy spikes outside merge_gap_ms should remain separate."""
        timeline = [
            _frame(0, "happy", 0.85),
            _frame(300, "happy", 0.90),  # 300ms gap > merge_gap_ms=150
        ]
        peaks = _detector(spike_threshold=0.80, merge_gap_ms=150).detect(
            _session(timeline)
        )
        spikes = [p for p in peaks if p["peak_type"] == "spike"]
        assert len(spikes) == 2

    def test_spike_duration_ms_is_zero(self):
        timeline = [_frame(500, "happy", 0.90)]
        peaks = _detector(spike_threshold=0.80).detect(_session(timeline))
        spikes = [p for p in peaks if p["peak_type"] == "spike"]
        assert spikes[0]["duration_ms"] == 0


# ---------------------------------------------------------------------------
# Transcript lookup tests
# ---------------------------------------------------------------------------

class TestTranscriptLookup:

    def test_transcript_segment_populated_from_annotations(self):
        timeline = [
            _frame(0, "neutral", 0.70),
            _frame(600, "happy", 0.85),
        ]
        annotations = [
            _annotation(0, "Before the punchline."),
            _annotation(600, "So I was at the DMV..."),
        ]
        peaks = _detector(spike_threshold=0.99).detect(
            _session(timeline, annotations)
        )
        onsets = [p for p in peaks if p["peak_type"] == "onset"]
        assert len(onsets) == 1
        assert onsets[0]["transcript_segment"] == "So I was at the DMV..."

    def test_transcript_segment_is_none_when_not_in_annotations(self):
        timeline = [
            _frame(0, "neutral", 0.70),
            _frame(600, "happy", 0.85),
        ]
        # No annotations entry for timestamp 600
        annotations = [_annotation(0, "Some text.")]
        peaks = _detector(spike_threshold=0.99).detect(
            _session(timeline, annotations)
        )
        onsets = [p for p in peaks if p["peak_type"] == "onset"]
        assert onsets[0]["transcript_segment"] is None

    def test_transcript_segment_none_when_no_annotations(self):
        timeline = [_frame(0, "neutral", 0.70), _frame(600, "happy", 0.85)]
        peaks = _detector(spike_threshold=0.99).detect(_session(timeline, []))
        for peak in peaks:
            assert peak["transcript_segment"] is None


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_empty_timeline_returns_empty_list(self):
        peaks = _detector().detect(_session([]))
        assert peaks == []

    def test_all_neutral_returns_empty_list(self):
        timeline = [
            _frame(i * 100, "neutral", 0.70) for i in range(10)
        ]
        peaks = _detector().detect(_session(timeline))
        assert peaks == []

    def test_all_no_face_returns_empty_list(self):
        timeline = [
            _frame(i * 100, "no_face", 0.0) for i in range(10)
        ]
        peaks = _detector().detect(_session(timeline))
        assert peaks == []

    def test_single_non_neutral_frame_no_baseline_produces_no_onset(self):
        """One non-neutral frame with no preceding frames → no onset, maybe spike."""
        timeline = [_frame(0, "happy", 0.70)]
        peaks = _detector(spike_threshold=0.99).detect(_session(timeline))
        onsets = [p for p in peaks if p["peak_type"] == "onset"]
        assert len(onsets) == 0

    def test_single_high_confidence_frame_produces_spike_only(self):
        timeline = [_frame(0, "happy", 0.90)]
        peaks = _detector(onset_window_ms=500, spike_threshold=0.80).detect(
            _session(timeline)
        )
        # No onset (no baseline window), but should produce a spike
        spikes = [p for p in peaks if p["peak_type"] == "spike"]
        assert len(spikes) == 1

    def test_detect_called_on_full_session_dict(self):
        """detect() works on a complete session dict, not just the timeline key."""
        session = {
            "session_id": "test-id",
            "video_source": "clip.mp4",
            "video_duration_ms": 5000,
            "viewer_id": "anon",
            "recorded_at": "2025-01-01T00:00:00+00:00",
            "emotion_timeline": [
                _frame(0, "neutral", 0.70),
                _frame(600, "happy", 0.85),
            ],
            "transcript_segments": [],
            "aligned_annotations": [],
        }
        peaks = _detector(spike_threshold=0.99).detect(session)
        assert isinstance(peaks, list)


# ---------------------------------------------------------------------------
# SessionAssembler integration tests
# ---------------------------------------------------------------------------

class TestSessionAssemblerIntegration:

    def test_annotated_peaks_key_present_in_assembled_session(self):
        assembler = SessionAssembler(
            video_source="clip.mp4",
            video_duration_ms=5000,
            session_id="fixed-id",
        )
        records = [
            EmotionRecord(
                timestamp_ms=0, emotion="neutral", confidence=0.70,
                face_detected=True, all_scores={},
            ),
            EmotionRecord(
                timestamp_ms=1500, emotion="happy", confidence=0.85,
                face_detected=True, all_scores={},
            ),
        ]
        session = assembler.assemble(records, [])
        assert "annotated_peaks" in session

    def test_annotated_peaks_is_a_list(self):
        assembler = SessionAssembler(
            video_source="clip.mp4",
            video_duration_ms=5000,
            session_id="fixed-id",
        )
        session = assembler.assemble([], [])
        assert isinstance(session["annotated_peaks"], list)

    def test_annotated_peaks_empty_for_all_neutral_session(self):
        assembler = SessionAssembler(
            video_source="clip.mp4",
            video_duration_ms=5000,
            session_id="fixed-id",
        )
        records = [
            EmotionRecord(
                timestamp_ms=i * 100, emotion="neutral", confidence=0.70,
                face_detected=True, all_scores={},
            )
            for i in range(10)
        ]
        session = assembler.assemble(records, [])
        assert session["annotated_peaks"] == []

    def test_phase4_tests_still_pass_with_new_key(self):
        """
        The Phase 4 schema tests check for a fixed set of top-level keys.
        Verify the new annotated_peaks key is present alongside the Phase 4 keys.
        """
        assembler = SessionAssembler(
            video_source="clip.mp4",
            video_duration_ms=5000,
            session_id="fixed-id",
        )
        session = assembler.assemble([], [])
        phase4_keys = {
            "session_id", "video_source", "video_duration_ms",
            "viewer_id", "recorded_at", "emotion_timeline",
            "transcript_segments", "aligned_annotations",
        }
        assert phase4_keys.issubset(set(session.keys()))
        assert "annotated_peaks" in session

    def test_custom_peak_detector_injected(self):
        """PeakDetector can be injected for testing with custom thresholds."""
        mock_detector = MagicMock()
        mock_detector.detect.return_value = [{"peak_type": "spike", "emotion": "happy"}]

        assembler = SessionAssembler(
            video_source="clip.mp4",
            video_duration_ms=5000,
            peak_detector=mock_detector,
        )
        session = assembler.assemble([], [])

        mock_detector.detect.assert_called_once()
        assert session["annotated_peaks"] == [{"peak_type": "spike", "emotion": "happy"}]
