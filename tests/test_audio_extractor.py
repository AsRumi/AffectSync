"""
Tests for pipeline.audio_extractor.AudioExtractor.

ffmpeg is mocked at the module level so these tests have no dependency
on the ffmpeg binary or any real video file. The only real filesystem
operations are creating a dummy input file and verifying output paths.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.audio_extractor import AudioExtractor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_video(tmp_path) -> Path:
    """A zero-byte file standing in for a real video — enough to pass exists()."""
    p = tmp_path / "clip.mp4"
    p.write_bytes(b"")
    return p


@pytest.fixture
def fake_wav(tmp_path) -> Path:
    """Simulates the WAV file that ffmpeg would write."""
    p = tmp_path / "clip.wav"
    p.write_bytes(b"\x00" * 1024)  # 1 KB of silence
    return p


# ---------------------------------------------------------------------------
# Helper: build a mock ffmpeg chain
# ---------------------------------------------------------------------------

def _make_ffmpeg_mock():
    """
    Return a mock that mimics the ffmpeg-python fluent API:
        ffmpeg.input(...).output(...).overwrite_output().run(quiet=True)
    Each method returns the same mock so chaining works.
    """
    mock = MagicMock()
    mock.input.return_value = mock
    mock.output.return_value = mock
    mock.overwrite_output.return_value = mock
    mock.run.return_value = (b"", b"")  # (stdout, stderr)
    return mock


# ---------------------------------------------------------------------------
# Lifecycle tests
# ---------------------------------------------------------------------------

class TestAudioExtractorInit:

    def test_wav_path_is_none_before_extract(self, fake_video, tmp_path):
        extractor = AudioExtractor(fake_video, output_dir=tmp_path)
        assert extractor.wav_path is None

    def test_raises_on_missing_video(self, tmp_path):
        extractor = AudioExtractor(tmp_path / "ghost.mp4", output_dir=tmp_path)
        with pytest.raises(FileNotFoundError, match="not found"):
            extractor.extract()


# ---------------------------------------------------------------------------
# Extract tests
# ---------------------------------------------------------------------------

class TestExtract:

    def test_extract_returns_wav_path(self, fake_video, tmp_path):
        ffmpeg_mock = _make_ffmpeg_mock()

        with patch("pipeline.audio_extractor.ffmpeg", ffmpeg_mock):
            # Simulate ffmpeg writing the wav file
            expected_wav = tmp_path / "clip.wav"
            expected_wav.write_bytes(b"\x00" * 512)

            extractor = AudioExtractor(fake_video, output_dir=tmp_path)
            result = extractor.extract()

        assert result == expected_wav
        assert extractor.wav_path == expected_wav

    def test_extract_creates_output_dir(self, fake_video, tmp_path):
        nested_dir = tmp_path / "a" / "b" / "c"
        ffmpeg_mock = _make_ffmpeg_mock()

        with patch("pipeline.audio_extractor.ffmpeg", ffmpeg_mock):
            # Simulate ffmpeg writing the file
            nested_dir.mkdir(parents=True, exist_ok=True)
            (nested_dir / "clip.wav").write_bytes(b"")

            extractor = AudioExtractor(fake_video, output_dir=nested_dir)
            extractor.extract()

        assert nested_dir.exists()

    def test_extract_calls_ffmpeg_with_correct_args(self, fake_video, tmp_path):
        ffmpeg_mock = _make_ffmpeg_mock()

        with patch("pipeline.audio_extractor.ffmpeg", ffmpeg_mock):
            (tmp_path / "clip.wav").write_bytes(b"")
            extractor = AudioExtractor(fake_video, output_dir=tmp_path, sample_rate=16000)
            extractor.extract()

        # input() should have been called with the video path
        ffmpeg_mock.input.assert_called_once_with(str(fake_video))

        # output() should include mono, 16kHz, no video
        _, kwargs = ffmpeg_mock.output.call_args
        assert kwargs.get("ac") == 1
        assert kwargs.get("ar") == 16000
        assert kwargs.get("acodec") == "pcm_s16le"

    def test_extract_raises_runtime_error_on_ffmpeg_failure(self, fake_video, tmp_path):
        import ffmpeg as real_ffmpeg

        ffmpeg_mock = _make_ffmpeg_mock()
        ffmpeg_mock.run.side_effect = real_ffmpeg.Error(
            "ffmpeg", b"", b"codec not found"
        )

        with patch("pipeline.audio_extractor.ffmpeg", ffmpeg_mock):
            extractor = AudioExtractor(fake_video, output_dir=tmp_path)
            with pytest.raises(RuntimeError, match="ffmpeg failed"):
                extractor.extract()

    def test_wav_name_matches_video_stem(self, fake_video, tmp_path):
        ffmpeg_mock = _make_ffmpeg_mock()

        with patch("pipeline.audio_extractor.ffmpeg", ffmpeg_mock):
            (tmp_path / "clip.wav").write_bytes(b"")
            extractor = AudioExtractor(fake_video, output_dir=tmp_path)
            wav = extractor.extract()

        assert wav.stem == fake_video.stem
        assert wav.suffix == ".wav"


# ---------------------------------------------------------------------------
# Cleanup tests
# ---------------------------------------------------------------------------

class TestCleanup:

    def test_cleanup_deletes_wav_file(self, fake_video, tmp_path):
        ffmpeg_mock = _make_ffmpeg_mock()

        with patch("pipeline.audio_extractor.ffmpeg", ffmpeg_mock):
            wav = tmp_path / "clip.wav"
            wav.write_bytes(b"\x00" * 512)

            extractor = AudioExtractor(fake_video, output_dir=tmp_path)
            extractor.extract()

        assert wav.exists()
        extractor.cleanup()
        assert not wav.exists()

    def test_cleanup_before_extract_is_safe(self, fake_video, tmp_path):
        """cleanup() with no WAV file should not raise."""
        extractor = AudioExtractor(fake_video, output_dir=tmp_path)
        extractor.cleanup()  # Should not raise

    def test_wav_path_is_none_after_cleanup(self, fake_video, tmp_path):
        ffmpeg_mock = _make_ffmpeg_mock()

        with patch("pipeline.audio_extractor.ffmpeg", ffmpeg_mock):
            (tmp_path / "clip.wav").write_bytes(b"")
            extractor = AudioExtractor(fake_video, output_dir=tmp_path)
            extractor.extract()

        extractor.cleanup()
        assert extractor.wav_path is None
