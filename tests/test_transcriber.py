"""
Tests for pipeline.transcriber.Transcriber.

All Whisper model calls are mocked — no model download or GPU required.
The mock mimics the dict structure that whisper.model.transcribe() returns.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.transcriber import Transcriber, RawSegment


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_whisper_result(segments: list[dict], language: str = "en") -> dict:
    """Build a mock return value matching whisper.transcribe() format."""
    return {
        "text": " ".join(s["text"] for s in segments),
        "segments": segments,
        "language": language,
    }


def _fake_segment(start: float, end: float, text: str) -> dict:
    return {"start": start, "end": end, "text": text, "id": 0}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_wav(tmp_path) -> Path:
    """A zero-byte file standing in for a real WAV."""
    p = tmp_path / "audio.wav"
    p.write_bytes(b"RIFF")  # Minimal header so it exists
    return p


@pytest.fixture
def mock_whisper_model():
    """A mock Whisper model whose transcribe() returns a canned result."""
    model = MagicMock()
    model.transcribe.return_value = _make_whisper_result([
        _fake_segment(0.0, 2.5, "Hello world."),
        _fake_segment(2.5, 5.0, "This is a test."),
    ])
    return model


# ---------------------------------------------------------------------------
# Model loading tests
# ---------------------------------------------------------------------------

class TestTranscriberInit:

    def test_model_not_loaded_on_init(self):
        transcriber = Transcriber()
        assert transcriber._model is None

    def test_model_loaded_lazily_on_first_transcribe(self, fake_wav, mock_whisper_model):
        with patch("pipeline.transcriber.whisper.load_model", return_value=mock_whisper_model):
            transcriber = Transcriber()
            assert transcriber._model is None
            transcriber.transcribe(fake_wav)
            assert transcriber._model is not None

    def test_model_loaded_only_once_across_calls(self, fake_wav, mock_whisper_model):
        with patch("pipeline.transcriber.whisper.load_model", return_value=mock_whisper_model) as mock_load:
            transcriber = Transcriber()
            transcriber.transcribe(fake_wav)
            transcriber.transcribe(fake_wav)
            assert mock_load.call_count == 1


# ---------------------------------------------------------------------------
# Transcription output tests
# ---------------------------------------------------------------------------

class TestTranscribe:

    def test_raises_on_missing_wav(self, tmp_path):
        transcriber = Transcriber()
        with pytest.raises(FileNotFoundError, match="not found"):
            transcriber.transcribe(tmp_path / "ghost.wav")

    def test_returns_list_of_raw_segments(self, fake_wav, mock_whisper_model):
        with patch("pipeline.transcriber.whisper.load_model", return_value=mock_whisper_model):
            transcriber = Transcriber()
            segments = transcriber.transcribe(fake_wav)

        assert isinstance(segments, list)
        assert all(isinstance(s, RawSegment) for s in segments)

    def test_segment_count_matches_whisper_output(self, fake_wav, mock_whisper_model):
        with patch("pipeline.transcriber.whisper.load_model", return_value=mock_whisper_model):
            transcriber = Transcriber()
            segments = transcriber.transcribe(fake_wav)

        assert len(segments) == 2

    def test_segment_timestamps_are_correct(self, fake_wav, mock_whisper_model):
        with patch("pipeline.transcriber.whisper.load_model", return_value=mock_whisper_model):
            transcriber = Transcriber()
            segments = transcriber.transcribe(fake_wav)

        assert segments[0].start_s == pytest.approx(0.0)
        assert segments[0].end_s == pytest.approx(2.5)
        assert segments[1].start_s == pytest.approx(2.5)
        assert segments[1].end_s == pytest.approx(5.0)

    def test_segment_text_is_stripped(self, fake_wav):
        model = MagicMock()
        model.transcribe.return_value = _make_whisper_result([
            _fake_segment(0.0, 1.0, "  leading and trailing spaces  "),
        ])
        with patch("pipeline.transcriber.whisper.load_model", return_value=model):
            transcriber = Transcriber()
            segments = transcriber.transcribe(fake_wav)

        assert segments[0].text == "leading and trailing spaces"

    def test_empty_segments_are_filtered_out(self, fake_wav):
        model = MagicMock()
        model.transcribe.return_value = _make_whisper_result([
            _fake_segment(0.0, 1.0, "Real speech."),
            _fake_segment(1.0, 2.0, "   "),   # Whitespace-only — should be skipped
            _fake_segment(2.0, 3.0, ""),       # Empty — should be skipped
            _fake_segment(3.0, 4.0, "More speech."),
        ])
        with patch("pipeline.transcriber.whisper.load_model", return_value=model):
            transcriber = Transcriber()
            segments = transcriber.transcribe(fake_wav)

        assert len(segments) == 2
        assert segments[0].text == "Real speech."
        assert segments[1].text == "More speech."

    def test_returns_empty_list_when_no_speech(self, fake_wav):
        model = MagicMock()
        model.transcribe.return_value = _make_whisper_result([])
        with patch("pipeline.transcriber.whisper.load_model", return_value=model):
            transcriber = Transcriber()
            segments = transcriber.transcribe(fake_wav)

        assert segments == []

    def test_transcribe_raises_runtime_error_on_whisper_failure(self, fake_wav, mock_whisper_model):
        mock_whisper_model.transcribe.side_effect = RuntimeError("CUDA out of memory")
        with patch("pipeline.transcriber.whisper.load_model", return_value=mock_whisper_model):
            transcriber = Transcriber()
            with pytest.raises(RuntimeError, match="Whisper transcription failed"):
                transcriber.transcribe(fake_wav)

    def test_fp16_is_false(self, fake_wav, mock_whisper_model):
        """Verify fp16=False is always passed to avoid CUDA-only fp16 errors on CPU."""
        with patch("pipeline.transcriber.whisper.load_model", return_value=mock_whisper_model):
            transcriber = Transcriber()
            transcriber.transcribe(fake_wav)

        _, kwargs = mock_whisper_model.transcribe.call_args
        assert kwargs.get("fp16") is False
