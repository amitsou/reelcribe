"""Tests for reelcribe.transcription module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from reelcribe.transcription import save_transcript, transcribe


class TestTranscribe:
    def test_raises_when_audio_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            transcribe(tmp_path / "missing.wav")

    def test_returns_transcript_text(self, tmp_path):
        audio = tmp_path / "audio.wav"
        audio.write_bytes(b"fake audio")

        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"text": "  Hello world  "}

        with patch.dict("sys.modules", {"whisper": MagicMock(load_model=lambda _: mock_model)}):
            result = transcribe(audio, model_name="base")

        assert result == "Hello world"
        mock_model.transcribe.assert_called_once_with(str(audio))

    def test_raises_import_error_when_whisper_missing(self, tmp_path):
        audio = tmp_path / "audio.wav"
        audio.write_bytes(b"fake audio")

        import sys
        original = sys.modules.pop("whisper", None)
        with patch.dict("sys.modules", {"whisper": None}):
            with pytest.raises(ImportError, match="openai-whisper"):
                transcribe(audio)
        if original is not None:
            sys.modules["whisper"] = original


class TestSaveTranscript:
    def test_writes_file(self, tmp_path):
        out = tmp_path / "sub" / "transcript.txt"
        result = save_transcript("Hello world", out)
        assert result == out
        assert out.read_text(encoding="utf-8") == "Hello world"

    def test_creates_parent_dirs(self, tmp_path):
        out = tmp_path / "a" / "b" / "c.txt"
        save_transcript("text", out)
        assert out.exists()
