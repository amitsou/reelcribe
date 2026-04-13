"""Speech-to-text using the local OpenAI Whisper Python package."""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def transcribe(audio_path: Path, model_name: str = "base") -> str:
    """Run Whisper on *audio_path* and return the transcript text.

    Parameters
    ----------
    audio_path:
        Path to a readable audio file (WAV is typical for this project).
    model_name:
        Whisper size name: ``tiny``, ``base``, ``small``, ``medium``, or ``large``.

    Returns
    -------
    str
        Stripped transcript text.

    Raises
    ------
    FileNotFoundError
        If *audio_path* is not a file.
    ImportError
        If the ``openai-whisper`` package is not installed.
    """
    try:
        import whisper  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "The 'openai-whisper' package is required for transcription.\n"
            "Install it with:  pip install openai-whisper"
        ) from exc

    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    logger.info("Loading Whisper model '%s' ...", model_name)
    model = whisper.load_model(model_name)

    logger.info("Transcribing %s ...", audio_path)
    result = model.transcribe(str(audio_path))

    text: str = result["text"].strip()
    logger.info("Transcription complete (%d chars)", len(text))
    return text


def save_transcript(text: str, output_path: Path) -> Path:
    """Write *text* to *output_path* using UTF-8 encoding.

    Parameters
    ----------
    text:
        Full transcript body.
    output_path:
        Destination path (parents are created if missing).

    Returns
    -------
    Path
        *output_path* after writing.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    logger.info("Transcript saved: %s", output_path)
    return output_path
