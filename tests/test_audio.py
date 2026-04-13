"""Tests for reelcribe.audio module."""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from reelcribe.audio import (
    SUPPORTED_VIDEO_EXTENSIONS,
    check_ffmpeg,
    extract_audio,
    find_video_files,
)


class TestCheckFfmpeg:
    def test_passes_when_ffmpeg_present(self):
        with patch("shutil.which", return_value="/usr/bin/ffmpeg"):
            check_ffmpeg()  # should not raise

    def test_raises_when_ffmpeg_missing(self):
        with patch("shutil.which", return_value=None):
            with pytest.raises(RuntimeError, match="ffmpeg was not found"):
                check_ffmpeg()


class TestExtractAudio:
    def test_raises_when_video_missing(self, tmp_path):
        with patch("shutil.which", return_value="/usr/bin/ffmpeg"):
            with pytest.raises(FileNotFoundError):
                extract_audio(tmp_path / "missing.mp4", tmp_path / "out.wav")

    def test_creates_output_dir_and_returns_path(self, tmp_path):
        video = tmp_path / "video.mp4"
        video.write_bytes(b"fake")
        out_dir = tmp_path / "sub" / "output"
        wav = out_dir / "video.wav"

        with patch("shutil.which", return_value="/usr/bin/ffmpeg"):
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0, stderr=b"")
                result = extract_audio(video, wav)

        assert result == wav
        assert out_dir.exists()
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert "ffmpeg" in cmd
        assert str(video) in cmd
        assert str(wav) in cmd

    def test_raises_on_ffmpeg_failure(self, tmp_path):
        video = tmp_path / "video.mp4"
        video.write_bytes(b"fake")
        wav = tmp_path / "out.wav"

        with patch("shutil.which", return_value="/usr/bin/ffmpeg"):
            with patch("subprocess.run") as mock_run:
                mock_run.side_effect = subprocess.CalledProcessError(
                    1, "ffmpeg", stderr=b"some error"
                )
                with pytest.raises(RuntimeError, match="ffmpeg failed"):
                    extract_audio(video, wav)


class TestFindVideoFiles:
    def test_finds_supported_extensions(self, tmp_path):
        for ext in SUPPORTED_VIDEO_EXTENSIONS:
            (tmp_path / f"clip{ext}").write_bytes(b"")
        # also create a file that should NOT be found
        (tmp_path / "notes.txt").write_bytes(b"")

        found = find_video_files(tmp_path)
        assert len(found) == len(SUPPORTED_VIDEO_EXTENSIONS)
        for f in found:
            assert f.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS

    def test_returns_empty_for_empty_dir(self, tmp_path):
        assert find_video_files(tmp_path) == []

    def test_results_are_sorted(self, tmp_path):
        names = ["c.mp4", "a.mp4", "b.mp4"]
        for n in names:
            (tmp_path / n).write_bytes(b"")
        found = find_video_files(tmp_path)
        assert [f.name for f in found] == sorted(names)
