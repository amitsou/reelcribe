"""FFmpeg-based extraction of WAV audio from video files."""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

SUPPORTED_VIDEO_EXTENSIONS = frozenset(
    {".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv", ".m4v"}
)


def check_ffmpeg() -> None:
    """Verify that the ``ffmpeg`` executable is available on ``PATH``.

    Raises
    ------
    RuntimeError
        If ``ffmpeg`` cannot be found.
    """
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg was not found on PATH. "
            "Install ffmpeg before using reelcribe.\n"
            "  Linux:   sudo apt install ffmpeg\n"
            "  macOS:   brew install ffmpeg\n"
            "  Windows: https://ffmpeg.org/download.html"
        )


def extract_audio(video_path: Path, output_path: Path) -> Path:
    """Decode *video_path* to mono 16 kHz PCM WAV at *output_path*.

    Uses ``ffmpeg`` with PCM signed 16-bit little-endian output, suitable
    for Whisper transcription.

    Parameters
    ----------
    video_path:
        Existing video file to read.
    output_path:
        Destination ``.wav`` path (parent directories are created if needed).

    Returns
    -------
    Path
        *output_path* after a successful run.

    Raises
    ------
    FileNotFoundError
        If *video_path* does not exist.
    RuntimeError
        If *ffmpeg* is missing or exits with an error.
    """
    check_ffmpeg()

    if not video_path.is_file():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        str(output_path),
    ]

    logger.debug("Running ffmpeg: %s", " ".join(cmd))

    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode(errors="replace")
        raise RuntimeError(
            f"ffmpeg failed for '{video_path}':\n{stderr}"
        ) from exc

    logger.debug("ffmpeg stderr: %s", result.stderr.decode(errors="replace"))
    logger.info("Audio extracted: %s", output_path)
    return output_path


def find_video_files(input_dir: Path) -> list[Path]:
    """List supported video files in *input_dir* (non-recursive).

    Files are sorted by name. Only direct children of *input_dir* are considered.

    Parameters
    ----------
    input_dir:
        Directory to scan.

    Returns
    -------
    list[Path]
        Paths to files whose suffix is in ``SUPPORTED_VIDEO_EXTENSIONS``.
    """
    video_files = [
        p
        for p in sorted(input_dir.iterdir())
        if p.is_file() and p.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS
    ]
    logger.debug("Found %d video file(s) in %s", len(video_files), input_dir)
    return video_files
