"""Tests for reelcribe.cli module."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from reelcribe.cli import _build_parser, main, process_file


class TestBuildParser:
    def test_required_args(self):
        parser = _build_parser()
        args = parser.parse_args(["-i", "/in", "-o", "/out"])
        assert args.input_dir == Path("/in")
        assert args.output_dir == Path("/out")
        assert args.mode == "transcribe"  # default

    def test_mode_choices(self):
        parser = _build_parser()
        for mode in ("audio", "transcribe", "full"):
            args = parser.parse_args(["-i", "/in", "-o", "/out", "--mode", mode])
            assert args.mode == mode

    def test_invalid_mode_exits(self):
        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["-i", "/in", "-o", "/out", "--mode", "invalid"])

    def test_skip_existing_flag(self):
        parser = _build_parser()
        args = parser.parse_args(["-i", "/in", "-o", "/out", "--skip-existing"])
        assert args.skip_existing is True


class TestMain:
    def test_returns_1_if_input_dir_missing(self, tmp_path):
        rc = main(["-i", str(tmp_path / "nope"), "-o", str(tmp_path / "out")])
        assert rc == 1

    def test_returns_0_if_no_videos_found(self, tmp_path):
        in_dir = tmp_path / "in"
        in_dir.mkdir()
        out_dir = tmp_path / "out"

        with patch("shutil.which", return_value="/usr/bin/ffmpeg"):
            rc = main(["-i", str(in_dir), "-o", str(out_dir)])
        assert rc == 0

    def test_returns_1_if_ffmpeg_missing(self, tmp_path):
        in_dir = tmp_path / "in"
        in_dir.mkdir()

        with patch("shutil.which", return_value=None):
            rc = main(["-i", str(in_dir), "-o", str(tmp_path / "out")])
        assert rc == 1

    def test_processes_video_files(self, tmp_path):
        in_dir = tmp_path / "in"
        in_dir.mkdir()
        (in_dir / "clip.mp4").write_bytes(b"fake")

        with patch("shutil.which", return_value="/usr/bin/ffmpeg"):
            with patch("reelcribe.cli.process_file") as mock_proc:
                rc = main(["-i", str(in_dir), "-o", str(tmp_path / "out"), "--mode", "audio"])

        assert rc == 0
        mock_proc.assert_called_once()


class TestProcessFile:
    def test_audio_mode_only_extracts_audio(self, tmp_path):
        video = tmp_path / "v.mp4"
        video.write_bytes(b"fake")

        with patch("reelcribe.cli.extract_audio") as mock_audio:
            with patch("reelcribe.cli.transcribe") as mock_transcribe:
                process_file(
                    video_path=video,
                    output_dir=tmp_path,
                    mode="audio",
                    whisper_model="base",
                    ollama_model="llama3",
                    ollama_url="http://localhost:11434/api/generate",
                    skip_existing=False,
                )
        mock_audio.assert_called_once()
        mock_transcribe.assert_not_called()

    def test_skip_existing_skips_wav(self, tmp_path):
        video = tmp_path / "v.mp4"
        video.write_bytes(b"fake")
        wav = tmp_path / "v.wav"
        wav.write_bytes(b"existing wav")

        with patch("reelcribe.cli.extract_audio") as mock_audio:
            process_file(
                video_path=video,
                output_dir=tmp_path,
                mode="audio",
                whisper_model="base",
                ollama_model="llama3",
                ollama_url="http://localhost:11434/api/generate",
                skip_existing=True,
            )
        mock_audio.assert_not_called()

    def test_transcribe_mode_saves_transcript(self, tmp_path):
        video = tmp_path / "v.mp4"
        video.write_bytes(b"fake")

        with patch("reelcribe.cli.extract_audio"):
            with patch("reelcribe.cli.transcribe", return_value="Hello"):
                with patch("reelcribe.cli.save_transcript") as mock_save:
                    with patch("reelcribe.cli.generate_title") as mock_title:
                        process_file(
                            video_path=video,
                            output_dir=tmp_path,
                            mode="transcribe",
                            whisper_model="base",
                            ollama_model="llama3",
                            ollama_url="http://localhost:11434/api/generate",
                            skip_existing=False,
                        )
        mock_save.assert_called_once()
        mock_title.assert_not_called()

    def test_full_mode_generates_title(self, tmp_path):
        video = tmp_path / "v.mp4"
        video.write_bytes(b"fake")

        with patch("reelcribe.cli.extract_audio"):
            with patch("reelcribe.cli.transcribe", return_value="Hello"):
                with patch("reelcribe.cli.save_transcript"):
                    with patch("reelcribe.cli.generate_title", return_value="Title") as mock_title:
                        with patch("reelcribe.cli.save_title") as mock_save_title:
                            process_file(
                                video_path=video,
                                output_dir=tmp_path,
                                mode="full",
                                whisper_model="base",
                                ollama_model="llama3",
                                ollama_url="http://localhost:11434/api/generate",
                                skip_existing=False,
                            )
        mock_title.assert_called_once_with(
            "Hello", model="llama3", ollama_url="http://localhost:11434/api/generate"
        )
        mock_save_title.assert_called_once()
