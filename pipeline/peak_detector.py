"""
Emotion peak detector.

Run a post-processing pass over emotion_timeline to identify three classes:
    onset, sustained, and spike emotions.

Input is the already-serialized emotion_timeline list of dicts (as it
appears in the session JSON), not raw EmotionRecord objects. This keeps
PeakDetector decoupled from the recording pipeline and lets it operate
on any session JSON loaded from disk.

Output is a list of peak dicts appended to the session under the key
"annotated_peaks". The transcript_segment field is populated by looking
up each peak's start_ms in the session's aligned_annotations index.
"""

import logging
import sys
from pathlib import Path
from typing import List, Optional

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from config import (
    PEAK_CONFIDENCE_THRESHOLD,
    PEAK_ONSET_WINDOW_MS,
    PEAK_SUSTAINED_MIN_DURATION_MS,
    PEAK_SPIKE_THRESHOLD,
    PEAK_MERGE_GAP_MS,
)

logger = logging.getLogger(__name__)

# Emotions that count as "neutral baseline" for onset detection.
# no_face and unknown are also baseline — they provide no emotional signal.
_BASELINE_EMOTIONS = {"neutral", "no_face", "unknown"}


class PeakDetector:
    """
    Detects emotional peaks in a session's emotion_timeline.

    Operates on the serialized timeline (list of dicts) so it can process
    both live sessions and sessions loaded from disk.

    Usage:
        detector = PeakDetector()
        peaks = detector.detect(session)
        session["annotated_peaks"] = peaks
    """

    def __init__(
        self,
        confidence_threshold: float = PEAK_CONFIDENCE_THRESHOLD,
        onset_window_ms: int = PEAK_ONSET_WINDOW_MS,
        sustained_min_ms: int = PEAK_SUSTAINED_MIN_DURATION_MS,
        spike_threshold: float = PEAK_SPIKE_THRESHOLD,
        merge_gap_ms: int = PEAK_MERGE_GAP_MS,
    ):
        self._confidence_threshold = confidence_threshold
        self._onset_window_ms = onset_window_ms
        self._sustained_min_ms = sustained_min_ms
        self._spike_threshold = spike_threshold
        self._merge_gap_ms = merge_gap_ms

    def detect(self, session: dict) -> List[dict]:
        """
        Run all three peak detectors over the session and return a combined,
        deduplicated, chronologically sorted list of peak dicts.

        Args:
            session: The assembled session dict containing at minimum
                     "emotion_timeline" and "aligned_annotations".

        Returns:
            List of peak dicts, each with keys:
                peak_type        - "onset" | "sustained" | "spike"
                emotion          - dominant emotion label
                start_ms         - timestamp of peak start
                end_ms           - timestamp of peak end (== start_ms for onset/spike)
                duration_ms      - end_ms - start_ms (0 for onset/spike)
                peak_confidence  - highest confidence value within the peak window
                transcript_segment - overlapping transcript text, or None
        """
        timeline = session.get("emotion_timeline", [])

        if not timeline:
            logger.info("Peak detection skipped — empty emotion timeline.")
            return []

        # Build a fast lookup: timestamp_ms -> transcript_segment text
        transcript_index = self._build_transcript_index(
            session.get("aligned_annotations", [])
        )

        onsets = self._detect_onsets(timeline, transcript_index)
        sustained = self._detect_sustained(timeline, transcript_index)
        spikes = self._detect_spikes(timeline, transcript_index)

        all_peaks = onsets + sustained + spikes
        all_peaks.sort(key=lambda p: p["start_ms"])

        logger.info(
            "Peak detection complete — %d onset, %d sustained, %d spike (%d total)",
            len(onsets),
            len(sustained),
            len(spikes),
            len(all_peaks),
        )
        return all_peaks

    # Onset detection
    def _detect_onsets(
        self,
        timeline: List[dict],
        transcript_index: dict,
    ) -> List[dict]:
        """
        Find frames where emotion transitions from a neutral baseline into
        a non-neutral emotion above the confidence threshold.

        For each candidate frame, look back PEAK_ONSET_WINDOW_MS and check
        that all frames in that window are baseline emotions. If yes, this
        frame is an onset.
        """
        peaks = []

        for i, frame in enumerate(timeline):
            emotion = frame.get("emotion", "unknown")
            confidence = frame.get("confidence", 0.0)

            # Must be non-baseline and above threshold
            if emotion in _BASELINE_EMOTIONS:
                continue
            if confidence < self._confidence_threshold:
                continue

            timestamp_ms = frame["timestamp_ms"]
            window_start = timestamp_ms - self._onset_window_ms

            # Check all preceding frames within the onset window
            baseline_window = [
                f for f in timeline
                if window_start <= f["timestamp_ms"] < timestamp_ms
            ]

            # Need at least one frame in the window to confirm a baseline existed
            if not baseline_window:
                continue

            all_baseline = all(
                f.get("emotion", "unknown") in _BASELINE_EMOTIONS
                for f in baseline_window
            )

            if all_baseline:
                peaks.append(self._make_peak(
                    peak_type="onset",
                    emotion=emotion,
                    start_ms=timestamp_ms,
                    end_ms=timestamp_ms,
                    peak_confidence=confidence,
                    transcript_index=transcript_index,
                ))

        return peaks

    # Sustained peak detection
    def _detect_sustained(
        self,
        timeline: List[dict],
        transcript_index: dict,
    ) -> List[dict]:
        """
        Find runs of consecutive frames sharing the same non-baseline emotion
        above confidence threshold, lasting at least PEAK_SUSTAINED_MIN_DURATION_MS.

        "Consecutive" here means the frames are adjacent in the timeline list.
        A single dropped frame (no_face) within an otherwise sustained run
        does NOT break the run — we allow one-frame interruptions to handle
        momentary detection failures gracefully.
        """
        peaks = []
        n = len(timeline)
        i = 0

        while i < n:
            frame = timeline[i]
            emotion = frame.get("emotion", "unknown")
            confidence = frame.get("confidence", 0.0)

            # Look for the start of a qualifying run
            if emotion in _BASELINE_EMOTIONS or confidence < self._confidence_threshold:
                i += 1
                continue

            # We have a qualifying frame — extend the run forward
            run_emotion = emotion
            run_start_ms = frame["timestamp_ms"]
            run_end_ms = frame["timestamp_ms"]
            run_peak_confidence = confidence
            skip_count = 0  # Consecutive non-qualifying frames tolerated

            j = i + 1
            while j < n:
                next_frame = timeline[j]
                next_emotion = next_frame.get("emotion", "unknown")
                next_confidence = next_frame.get("confidence", 0.0)

                qualifies = (
                    next_emotion == run_emotion
                    and next_confidence >= self._confidence_threshold
                )
                is_interruption = next_emotion in _BASELINE_EMOTIONS

                if qualifies:
                    run_end_ms = next_frame["timestamp_ms"]
                    run_peak_confidence = max(run_peak_confidence, next_confidence)
                    skip_count = 0
                    j += 1
                elif is_interruption and skip_count == 0:
                    # Allow one baseline interruption (e.g., momentary no_face)
                    skip_count += 1
                    j += 1
                else:
                    break

            duration_ms = run_end_ms - run_start_ms

            if duration_ms >= self._sustained_min_ms:
                peaks.append(self._make_peak(
                    peak_type="sustained",
                    emotion=run_emotion,
                    start_ms=run_start_ms,
                    end_ms=run_end_ms,
                    peak_confidence=run_peak_confidence,
                    transcript_index=transcript_index,
                ))

            # Advance past the run we just processed
            i = j

        return peaks

    # Spike detection
    def _detect_spikes(
        self,
        timeline: List[dict],
        transcript_index: dict,
    ) -> List[dict]:
        """
        Find individual frames where confidence in a non-baseline emotion
        exceeds PEAK_SPIKE_THRESHOLD, regardless of surrounding context.

        Nearby spikes of the same emotion within PEAK_MERGE_GAP_MS are
        merged into a single spike entry keyed to the highest-confidence
        frame in that cluster.
        """
        raw_spikes = []

        for frame in timeline:
            emotion = frame.get("emotion", "unknown")
            confidence = frame.get("confidence", 0.0)

            if emotion in _BASELINE_EMOTIONS:
                continue
            if confidence < self._spike_threshold:
                continue

            raw_spikes.append({
                "emotion": emotion,
                "timestamp_ms": frame["timestamp_ms"],
                "confidence": confidence,
            })

        # Merge nearby spikes of the same emotion
        merged = self._merge_spikes(raw_spikes)

        return [
            self._make_peak(
                peak_type="spike",
                emotion=s["emotion"],
                start_ms=s["timestamp_ms"],
                end_ms=s["timestamp_ms"],
                peak_confidence=s["confidence"],
                transcript_index=transcript_index,
            )
            for s in merged
        ]

    def _merge_spikes(self, raw_spikes: List[dict]) -> List[dict]:
        """
        Merge adjacent spikes of the same emotion within PEAK_MERGE_GAP_MS
        into a single representative spike (the highest-confidence one).
        """
        if not raw_spikes:
            return []

        merged = []
        cluster = [raw_spikes[0]]

        for spike in raw_spikes[1:]:
            prev = cluster[-1]
            gap = spike["timestamp_ms"] - prev["timestamp_ms"]

            if spike["emotion"] == prev["emotion"] and gap <= self._merge_gap_ms:
                cluster.append(spike)
            else:
                merged.append(max(cluster, key=lambda s: s["confidence"]))
                cluster = [spike]

        merged.append(max(cluster, key=lambda s: s["confidence"]))
        return merged

    def _build_transcript_index(self, aligned_annotations: List[dict]) -> dict:
        """
        Build a dict mapping timestamp_ms -> transcript_segment text (or None)
        from the session's aligned_annotations for O(1) lookups during detection.
        """
        return {
            entry["timestamp_ms"]: entry.get("transcript_segment")
            for entry in aligned_annotations
        }

    def _make_peak(
        self,
        peak_type: str,
        emotion: str,
        start_ms: int,
        end_ms: int,
        peak_confidence: float,
        transcript_index: dict,
    ) -> dict:
        """Construct a peak dict with the standard schema."""
        return {
            "peak_type": peak_type,
            "emotion": emotion,
            "start_ms": start_ms,
            "end_ms": end_ms,
            "duration_ms": end_ms - start_ms,
            "peak_confidence": round(peak_confidence, 4),
            "transcript_segment": transcript_index.get(start_ms),
        }
