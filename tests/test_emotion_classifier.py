"""
Unit tests for pipeline.emotion_classifier.EmotionClassifier.

All DeepFace calls are mocked — no model download or GPU required.
"""

import numpy as np
import pytest
from unittest.mock import patch

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.emotion_classifier import EmotionClassifier


@pytest.fixture
def classifier():
    return EmotionClassifier(confidence_threshold=0.30)


@pytest.fixture
def fake_face():
    """A 48x48 dummy face crop."""
    return np.random.randint(0, 255, (48, 48, 3), dtype=np.uint8)


def _mock_deepface_result(dominant: str, scores: dict):
    """Build a mock return value matching DeepFace.analyze() format."""
    return [{"dominant_emotion": dominant, "emotion": scores}]


class TestEmotionClassifier:
    def test_classify_returns_dominant_emotion(self, classifier, fake_face):
        mock_result = _mock_deepface_result(
            "happy", {"happy": 85.0, "neutral": 10.0, "sad": 5.0}
        )
        with patch("pipeline.emotion_classifier.DeepFace.analyze", return_value=mock_result):
            emotion, confidence, all_scores = classifier.classify(fake_face)
            assert emotion == "happy"
            assert confidence == 0.85
            assert "happy" in all_scores

    def test_classify_falls_back_to_neutral_below_threshold(self, classifier, fake_face):
        mock_result = _mock_deepface_result(
            "sad", {"sad": 20.0, "neutral": 50.0, "happy": 30.0}
        )
        with patch("pipeline.emotion_classifier.DeepFace.analyze", return_value=mock_result):
            emotion, confidence, _ = classifier.classify(fake_face)
            # sad=0.20 < threshold=0.30, so should fall back to neutral
            assert emotion == "neutral"
            assert confidence == 0.50

    def test_classify_normalizes_scores_to_zero_one(self, classifier, fake_face):
        mock_result = _mock_deepface_result(
            "surprised", {"surprised": 72.5, "happy": 15.0, "neutral": 12.5}
        )
        with patch("pipeline.emotion_classifier.DeepFace.analyze", return_value=mock_result):
            _, _, all_scores = classifier.classify(fake_face)
            assert all_scores["surprised"] == 0.725
            assert all_scores["happy"] == 0.15
            assert all_scores["neutral"] == 0.125

    def test_classify_handles_exception_gracefully(self, classifier, fake_face):
        with patch(
            "pipeline.emotion_classifier.DeepFace.analyze",
            side_effect=ValueError("Model failed"),
        ):
            emotion, confidence, all_scores = classifier.classify(fake_face)
            assert emotion == "unknown"
            assert confidence == 0.0
            assert all_scores == {}

    def test_classify_returns_all_seven_labels(self, classifier, fake_face):
        full_scores = {
            "angry": 2.0, "disgusted": 1.0, "fearful": 3.0,
            "happy": 70.0, "sad": 5.0, "surprised": 4.0, "neutral": 15.0,
        }
        mock_result = _mock_deepface_result("happy", full_scores)
        with patch("pipeline.emotion_classifier.DeepFace.analyze", return_value=mock_result):
            _, _, all_scores = classifier.classify(fake_face)
            assert len(all_scores) == 7
            assert sum(all_scores.values()) == pytest.approx(1.0, abs=0.01)

    def test_warm_up_sets_flag(self, classifier):
        with patch("pipeline.emotion_classifier.DeepFace.analyze", return_value=[{}]):
            classifier.warm_up()
            assert classifier._warmed_up is True

    def test_warm_up_only_runs_once(self, classifier):
        with patch("pipeline.emotion_classifier.DeepFace.analyze") as mock_analyze:
            mock_analyze.return_value = [{}]
            classifier.warm_up()
            classifier.warm_up()
            assert mock_analyze.call_count == 1
