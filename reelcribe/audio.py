"""Audio extraction module – converts video files to WAV using ffmpeg."""

import logging
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv", ".m4v"}


def check_ffmpeg() -> None:
    """Raise RuntimeError if ffmpeg is not found on PATH."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg was not found on PATH. "
            "Please install ffmpeg before using reelcribe.\n"
            "  Linux:   sudo apt install ffmpeg\n"
            "  macOS:   brew install ffmpeg\n"
            "  Windows: https://ffmpeg.org/download.html"
        )


def extract_audio(video_path: Path, output_path: Path) -> Path:
    """Extract audio from *video_path* and save as a WAV file at *output_path*.

    Parameters
    ----------
    video_path:
        Path to the source video file.
    output_path:
        Destination path for the extracted WAV file.

    Returns
    -------
    Path
        The path to the written WAV file.

    Raises
    ------
    RuntimeError
        If ffmpeg is not installed or the extraction fails.
    FileNotFoundError
        If *video_path* does not exist.
    """
    check_ffmpeg()

    if not video_path.is_file():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",                  # overwrite without asking
        "-i", str(video_path),
        "-vn",                 # no video
        "-acodec", "pcm_s16le",
        "-ar", "16000",        # 16 kHz sample rate (optimal for Whisper)
        "-ac", "1",            # mono
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
    """Return a sorted list of supported video files inside *input_dir*."""
    video_files = [
        p
        for p in sorted(input_dir.iterdir())
        if p.is_file() and p.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS
    ]
    logger.debug("Found %d video file(s) in %s", len(video_files), input_dir)
    return video_files
